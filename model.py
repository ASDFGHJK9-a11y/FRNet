
from typing import Optional, Tuple, Union, Dict
import math
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F

from transformer import TransformerEncoder
from model_config import get_config


def make_divisible(
    v: Union[float, int],
    divisor: Optional[int] = 8,
    min_value: Optional[Union[float, int]] = None,
) -> Union[float, int]:

    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

class ContrastiveLoss(nn.Module):
    def __init__(self, fixed_margin: float = 0.5, lambda_weight: float = 0.6):
        super(ContrastiveLoss, self).__init__()
        self.fixed_margin = fixed_margin
        self.lambda_weight = lambda_weight

    def compute_adaptive_margin(self, features: Tensor) -> float:
        n = features.size(0)
        sim_matrix = F.cosine_similarity(features.unsqueeze(1), features.unsqueeze(0), dim=2)
        # 计算 adaptive margin
        margin = sim_matrix.sum() / (n * n)
        return margin

    def compute_mixed_margin(self, adaptive_margin: float) -> float:
        # 计算混合 Margin
        mixed_margin = self.lambda_weight * self.fixed_margin + (1 - self.lambda_weight) * adaptive_margin
        return mixed_margin

    def forward(self, features, labels):
        n = features.size(0)
        sim_matrix = F.cosine_similarity(features.unsqueeze(1), features.unsqueeze(0), dim=2)

        labels_matrix = labels.unsqueeze(1) == labels.unsqueeze(0)

        positive_mask = labels_matrix.float()
        negative_mask = 1 - positive_mask

        adaptive_margin = self.compute_adaptive_margin(features)
        mixed_margin = self.compute_mixed_margin(adaptive_margin)

        positive_loss = (1 - sim_matrix) * positive_mask
        negative_loss = torch.max(sim_matrix - mixed_margin, torch.tensor(0.0).to(sim_matrix.device)) * negative_mask

        loss = positive_loss.sum() / positive_mask.sum() + negative_loss.sum() / negative_mask.sum()
        return loss


def global_median_pooling(x):
    # 对输入特征图进行全局中值池化操作
    median_pooled = torch.median(x.view(x.size(0), x.size(1), -1), dim=2)[0]
    median_pooled = median_pooled.view(x.size(0), x.size(1), 1, 1)
    return median_pooled


class ChannelAttention(nn.Module):
    def __init__(self, input_channels, internal_neurons):
        super(ChannelAttention, self).__init__()
        self.fc1 = nn.Conv2d(in_channels=input_channels, out_channels=internal_neurons, kernel_size=1, stride=1,
                             bias=True)
        self.fc2 = nn.Conv2d(in_channels=internal_neurons, out_channels=input_channels, kernel_size=1, stride=1,
                             bias=True)

    def forward(self, inputs):
        avg_pool = F.adaptive_avg_pool2d(inputs, output_size=(1, 1))
        max_pool = F.adaptive_max_pool2d(inputs, output_size=(1, 1))
        median_pool = global_median_pooling(inputs)


        def process_pool(pool):
            out = self.fc1(pool)
            out = F.leaky_relu(out, inplace=True)
            out = self.fc2(out)
            return torch.sigmoid(out)

        avg_out = process_pool(avg_pool)
        max_out = process_pool(max_pool)
        median_out = process_pool(median_pool)

        # 合并三种注意力机制的结果
        return avg_out + max_out + median_out


class PyramidFeatures(nn.Module):
    def __init__(self, in_channels, levels=[1, 3, 5, 7]):
        super(PyramidFeatures, self).__init__()
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels, in_channels, kernel_size=(level, level), padding=level // 2, groups=in_channels)
            for level in levels
        ])

    def forward(self, x):
        features = [conv(x) for conv in self.convs]
        return torch.cat(features, dim=1)


class MEFA(nn.Module):
    def __init__(self, in_channels, out_channels, channel_attention_reduce=4):
        super(MEFA, self).__init__()
        assert in_channels == out_channels, "Input and output channels must be the same"
        self.channel_attention = ChannelAttention(in_channels, in_channels // channel_attention_reduce)
        self.pyramid_features = PyramidFeatures(in_channels)
        self.initial_depth_conv = nn.Conv2d(in_channels, in_channels, kernel_size=5, padding=2, groups=in_channels)
        self.final_conv = nn.Conv2d(in_channels * (1 + 4), out_channels, kernel_size=1)
        self.act = nn.Mish()

    def forward(self, inputs):
        inputs = self.act(inputs)
        attention = self.channel_attention(inputs)
        inputs = attention * inputs
        initial_out = self.initial_depth_conv(inputs)
        pyramid_out = self.pyramid_features(initial_out)
        combined_features = torch.cat([initial_out, pyramid_out], dim=1)
        out = self.final_conv(combined_features)
        out = self.act(out)
        return out
class ConvLayer(nn.Module):


    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Optional[Union[int, Tuple[int, int]]] = 1,
        groups: Optional[int] = 1,
        bias: Optional[bool] = False,
        use_norm: Optional[bool] = True,
        use_act: Optional[bool] = True,
    ) -> None:
        super().__init__()

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)

        if isinstance(stride, int):
            stride = (stride, stride)

        assert isinstance(kernel_size, Tuple)
        assert isinstance(stride, Tuple)

        padding = (
            int((kernel_size[0] - 1) / 2),
            int((kernel_size[1] - 1) / 2),
        )

        block = nn.Sequential()

        conv_layer = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            groups=groups,
            padding=padding,
            bias=bias
        )

        block.add_module(name="conv", module=conv_layer)

        if use_norm:
            norm_layer = nn.BatchNorm2d(num_features=out_channels, momentum=0.1)
            block.add_module(name="norm", module=norm_layer)

        if use_act:
            act_layer = nn.SiLU()
            block.add_module(name="act", module=act_layer)

        self.block = block

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)

class InvertedResidual(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        expand_ratio: Union[int, float],
        skip_connection: Optional[bool] = True,
    ) -> None:
        assert stride in [1, 2]
        hidden_dim = make_divisible(int(round(in_channels * expand_ratio)), 8)

        super().__init__()

        block = nn.Sequential()
        if expand_ratio != 1:
            block.add_module(
                name="exp_1x1",
                module=ConvLayer(
                    in_channels=in_channels,
                    out_channels=hidden_dim,
                    kernel_size=1
                ),
            )

        block.add_module(
            name="conv_3x3",
            module=ConvLayer(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                stride=stride,
                kernel_size=3,
                groups=hidden_dim
            ),
        )

        block.add_module(
            name="red_1x1",
            module=ConvLayer(
                in_channels=hidden_dim,
                out_channels=out_channels,
                kernel_size=1,
                use_act=False,
                use_norm=True,
            ),
        )

        self.block = block
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.exp = expand_ratio
        self.stride = stride
        self.use_res_connect = (
            self.stride == 1 and in_channels == out_channels and skip_connection
        )

    def forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        if self.use_res_connect:
            return x + self.block(x)
        else:
            return self.block(x)


class MobileViTBlock(nn.Module):


    def __init__(
        self,
        in_channels: int,
        transformer_dim: int,
        ffn_dim: int,
        n_transformer_blocks: int = 2,
        head_dim: int = 32,
        attn_dropout: float = 0.0,
        dropout: float = 0.0,
        ffn_dropout: float = 0.0,
        patch_h: int = 8,
        patch_w: int = 8,
        conv_ksize: Optional[int] = 3,
        *args,
        **kwargs
    ) -> None:
        super().__init__()

        conv_3x3_in = ConvLayer(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=conv_ksize,
            stride=1
        )
        conv_1x1_in = ConvLayer(
            in_channels=in_channels,
            out_channels=transformer_dim,
            kernel_size=1,
            stride=1,
            use_norm=False,
            use_act=False
        )

        conv_1x1_out = ConvLayer(
            in_channels=transformer_dim,
            out_channels=in_channels,
            kernel_size=1,
            stride=1
        )
        conv_3x3_out = ConvLayer(
            in_channels=2 * in_channels,
            out_channels=in_channels,
            kernel_size=conv_ksize,
            stride=1
        )

        self.local_rep = nn.Sequential()
        self.local_rep.add_module(name="conv_3x3", module=conv_3x3_in)
        self.local_rep.add_module(name="conv_1x1", module=conv_1x1_in)

        assert transformer_dim % head_dim == 0
        num_heads = transformer_dim // head_dim

        global_rep = [
            TransformerEncoder(
                embed_dim=transformer_dim,
                ffn_latent_dim=ffn_dim,
                num_heads=num_heads,
                attn_dropout=attn_dropout,
                dropout=dropout,
                ffn_dropout=ffn_dropout
            )
            for _ in range(n_transformer_blocks)
        ]
        global_rep.append(nn.LayerNorm(transformer_dim))
        self.global_rep = nn.Sequential(*global_rep)

        self.conv_proj = conv_1x1_out
        self.fusion = conv_3x3_out

        self.patch_h = patch_h
        self.patch_w = patch_w
        self.patch_area = self.patch_w * self.patch_h

        self.cnn_in_dim = in_channels
        self.cnn_out_dim = transformer_dim
        self.n_heads = num_heads
        self.ffn_dim = ffn_dim
        self.dropout = dropout
        self.attn_dropout = attn_dropout
        self.ffn_dropout = ffn_dropout
        self.n_blocks = n_transformer_blocks
        self.conv_ksize = conv_ksize

    def unfolding(self, x: Tensor) -> Tuple[Tensor, Dict]:
        patch_w, patch_h = self.patch_w, self.patch_h
        patch_area = patch_w * patch_h
        batch_size, in_channels, orig_h, orig_w = x.shape

        new_h = int(math.ceil(orig_h / self.patch_h) * self.patch_h)
        new_w = int(math.ceil(orig_w / self.patch_w) * self.patch_w)

        interpolate = False
        if new_w != orig_w or new_h != orig_h:
            # Note: Padding can be done, but then it needs to be handled in attention function.
            x = F.interpolate(x, size=(new_h, new_w), mode="bilinear", align_corners=False)
            interpolate = True

        # number of patches along width and height
        num_patch_w = new_w // patch_w  # n_w
        num_patch_h = new_h // patch_h  # n_h
        num_patches = num_patch_h * num_patch_w  # N

        # [B, C, H, W] -> [B * C * n_h, p_h, n_w, p_w]
        x = x.reshape(batch_size * in_channels * num_patch_h, patch_h, num_patch_w, patch_w)
        # [B * C * n_h, p_h, n_w, p_w] -> [B * C * n_h, n_w, p_h, p_w]
        x = x.transpose(1, 2)
        # [B * C * n_h, n_w, p_h, p_w] -> [B, C, N, P] where P = p_h * p_w and N = n_h * n_w
        x = x.reshape(batch_size, in_channels, num_patches, patch_area)
        # [B, C, N, P] -> [B, P, N, C]
        x = x.transpose(1, 3)
        # [B, P, N, C] -> [BP, N, C]
        x = x.reshape(batch_size * patch_area, num_patches, -1)

        info_dict = {
            "orig_size": (orig_h, orig_w),
            "batch_size": batch_size,
            "interpolate": interpolate,
            "total_patches": num_patches,
            "num_patches_w": num_patch_w,
            "num_patches_h": num_patch_h,
        }

        return x, info_dict

    def folding(self, x: Tensor, info_dict: Dict) -> Tensor:
        n_dim = x.dim()
        assert n_dim == 3, "Tensor should be of shape BPxNxC. Got: {}".format(
            x.shape
        )
        # [BP, N, C] --> [B, P, N, C]
        x = x.contiguous().view(
            info_dict["batch_size"], self.patch_area, info_dict["total_patches"], -1
        )

        batch_size, pixels, num_patches, channels = x.size()
        num_patch_h = info_dict["num_patches_h"]
        num_patch_w = info_dict["num_patches_w"]


        x = x.transpose(1, 3)

        x = x.reshape(batch_size * channels * num_patch_h, num_patch_w, self.patch_h, self.patch_w)

        x = x.transpose(1, 2)

        x = x.reshape(batch_size, channels, num_patch_h * self.patch_h, num_patch_w * self.patch_w)
        if info_dict["interpolate"]:
            x = F.interpolate(
                x,
                size=info_dict["orig_size"],
                mode="bilinear",
                align_corners=False,
            )
        return x

    def forward(self, x: Tensor) -> Tensor:
        res = x

        fm = self.local_rep(x)

        # convert feature map to patches
        patches, info_dict = self.unfolding(fm)

        # learn global representations
        for transformer_layer in self.global_rep:
            patches = transformer_layer(patches)


        fm = self.folding(x=patches, info_dict=info_dict)

        fm = self.conv_proj(fm)

        fm = self.fusion(torch.cat((res, fm), dim=1))
        return fm


class MobileViT(nn.Module):
    def __init__(self, model_cfg: Dict, num_classes: int = 23, lambda_contrastive: float = 1.0):
        super().__init__()

        image_channels = 4
        out_channels = 16

        self.conv_1 = ConvLayer(
            in_channels=image_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=2
        )


        self.layer_1, out_channels = self._make_layer(input_channel=out_channels, cfg=model_cfg["layer1"])





        self.layer_2, out_channels = self._make_layer(input_channel=out_channels, cfg=model_cfg["layer2"])


        self.layer_3, out_channels = self._make_layer(input_channel=out_channels, cfg=model_cfg["layer3"])
        self.mefa_1 = MEFA(in_channels=out_channels, out_channels=out_channels)

        self.layer_4, out_channels = self._make_layer(input_channel=out_channels, cfg=model_cfg["layer4"])

        self.layer_5, out_channels = self._make_layer(input_channel=out_channels, cfg=model_cfg["layer5"])



        exp_channels = min(model_cfg["last_layer_exp_factor"] * out_channels, 960)

        self.conv_1x1_exp = ConvLayer(
            in_channels=out_channels,
            out_channels=exp_channels,
            kernel_size=1
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(p=model_cfg["cls_dropout"]) if 0.0 < model_cfg["cls_dropout"] < 1.0 else nn.Identity(),
            nn.Linear(in_features=exp_channels, out_features=num_classes)
        )

        self.contrastive_loss = ContrastiveLoss()  # 对比损失实例
        self.lambda_contrastive = lambda_contrastive  # lambda值

        # 初始化权重
        self.apply(self.init_parameters)
    def _make_layer(self, input_channel, cfg: Dict) -> Tuple[nn.Sequential, int]:
        block_type = cfg.get("block_type", "mobilevit")
        if block_type.lower() == "mobilevit":
            return self._make_mit_layer(input_channel=input_channel, cfg=cfg)
        else:
            return self._make_mobilenet_layer(input_channel=input_channel, cfg=cfg)

    @staticmethod
    def _make_mobilenet_layer(input_channel: int, cfg: Dict) -> Tuple[nn.Sequential, int]:
        output_channels = cfg.get("out_channels")
        num_blocks = cfg.get("num_blocks", 2)
        expand_ratio = cfg.get("expand_ratio", 4)
        block = []

        for i in range(num_blocks):
            stride = cfg.get("stride", 1) if i == 0 else 1

            layer = InvertedResidual(
                in_channels=input_channel,
                out_channels=output_channels,
                stride=stride,
                expand_ratio=expand_ratio
            )
            block.append(layer)
            input_channel = output_channels

        return nn.Sequential(*block), input_channel

    @staticmethod
    def _make_mit_layer(input_channel: int, cfg: Dict) -> [nn.Sequential, int]:
        stride = cfg.get("stride", 1)
        block = []

        if stride == 2:
            layer = InvertedResidual(
                in_channels=input_channel,
                out_channels=cfg.get("out_channels"),
                stride=stride,
                expand_ratio=cfg.get("mv_expand_ratio", 4)
            )

            block.append(layer)
            input_channel = cfg.get("out_channels")

        transformer_dim = cfg["transformer_channels"]
        ffn_dim = cfg.get("ffn_dim")
        num_heads = cfg.get("num_heads", 4)
        head_dim = transformer_dim // num_heads

        if transformer_dim % head_dim != 0:
            raise ValueError("Transformer input dimension should be divisible by head dimension. "
                             "Got {} and {}.".format(transformer_dim, head_dim))

        block.append(MobileViTBlock(
            in_channels=input_channel,
            transformer_dim=transformer_dim,
            ffn_dim=ffn_dim,
            n_transformer_blocks=cfg.get("transformer_blocks", 1),
            patch_h=cfg.get("patch_h", 2),
            patch_w=cfg.get("patch_w", 2),
            dropout=cfg.get("dropout", 0.1),
            ffn_dropout=cfg.get("ffn_dropout", 0.0),
            attn_dropout=cfg.get("attn_dropout", 0.1),
            head_dim=head_dim,
            conv_ksize=3
        ))

        return nn.Sequential(*block), input_channel

    @staticmethod
    def init_parameters(m):
        if isinstance(m, nn.Conv2d):
            if m.weight is not None:
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            if m.weight is not None:
                nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.Linear,)):
            if m.weight is not None:
                nn.init.trunc_normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        else:
            pass

    def forward(self, x: Tensor, return_features: bool = False, targets: Tensor = None) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        x = self.conv_1(x)

        x = self.layer_1(x)



        x = self.layer_2(x)


        x = self.layer_3(x)

        x = self.mefa_1(x)
        x = self.layer_4(x)

        x = self.layer_5(x)
        x = self.conv_1x1_exp(x)
        if return_features:
            return x
        # 计算 logits
        logits = self.classifier(x)

        if targets is not None:
            # 计算交叉熵损失
            cross_entropy_loss = F.cross_entropy(logits, targets)

            # 计算对比损失
            contrastive_loss = self.contrastive_loss(x, targets)

            # 总损失
            total_loss = cross_entropy_loss + self.lambda_contrastive * contrastive_loss
            return logits, total_loss
        else:
            return logits


def mobile_vit_xx_small(num_classes: int = 23):

    config = get_config("xx_small")
    m = MobileViT(config, num_classes=num_classes)
    return m
