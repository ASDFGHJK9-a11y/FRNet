import os
import argparse
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchsummary import summary
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, f1_score, roc_auc_score, accuracy_score, roc_curve
from sklearn.manifold import TSNE
from my_dataset import MyDataSet
from model import mobile_vit_xx_small as create_model
from utils import read_split_data, train_one_epoch, evaluate

colors = [
    "#91B997", "#E16A3F", "#FDDC8D", "#A6DCC5", "#D5006D", "#F9B4D1",  # 清新绿色、红色、黄色、灰蓝色、深粉色、粉色
    "#B8CCE4", "#76ADD4", "#FF6347", "#D4D6C0", "#FF8C00", "#FF4359",  # 浅蓝色、天蓝色、番茄红、米色、橙色、鲜红
    "#379A2E", "#EEEEBC", "#8B0000", "#0079FF", "#00ECC2", "#1EF7FC",  # 深绿、浅黄、紫色、深蓝、青绿色、天蓝色
    "#4B0082", "#8A2BE2", "#A5B8B6", "#000000", "#FFD700"  # 靛蓝色、蓝紫色、灰绿色、橙色、深粉色
]


def extract_features(model, data_loader, device):
    """ 从模型中提取特征 """
    model.eval()
    all_features = []
    all_labels = []

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            # 通过传入 return_features=True 获取特征
            features = model(images, return_features=True)
            all_features.append(features.cpu().numpy())  # 提取特征
            all_labels.append(labels.cpu().numpy())  # 提取标签

    all_features = np.concatenate(all_features, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    return all_features, all_labels


def plot_tsne(features, labels, colors, output_path="tsne_plot.png", perplexity=5):
    """ 绘制带边框的实心 t-SNE 图，并设置困惑度（不添加图例） """

    # 如果 features 是 4D 张量 (如 (batch_size, channels, height, width)), 先展平为 2D 数组
    if isinstance(features, torch.Tensor) and len(features.shape) == 4:
        features = torch.flatten(features, start_dim=1)  # 展平为 (batch_size, channels * height * width)

    # 如果是 numpy 数组，确保是 2D
    elif isinstance(features, np.ndarray) and len(features.shape) == 4:
        features = features.reshape(features.shape[0], -1)  # 展平为 (batch_size, channels * height * width)

    # 确保 features 是 2D 数组
    assert len(features.shape) == 2, "Features should be a 2D array after flattening."

    # 使用 t-SNE 进行降维，并设置困惑度
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    tsne_features = tsne.fit_transform(features)

    # 绘制 t-SNE 图
    plt.figure(figsize=(10, 8))
    for i in range(len(colors)):
        # 对每个类别绘制一个实心散点图
        plt.scatter(tsne_features[labels == i, 0], tsne_features[labels == i, 1],
                    color=colors[i], alpha=0.6, s=20, marker='o')  # marker='o' 确保是实心点

    # 添加边框设置
    ax = plt.gca()  # 获取当前坐标轴
    ax.spines['top'].set_color('black')
    ax.spines['right'].set_color('black')
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_color('black')

    ax.spines['top'].set_linewidth(1.5)
    ax.spines['right'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['left'].set_linewidth(1.5)


    ax.set_xticks([])
    ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(output_path, dpi=1200)
    plt.close()


def plot_confusion_matrix(cm, class_names, output_path="confusion_matrix.png"):
    """ 绘制混淆矩阵并保存为图片 """
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names,
                cbar=False, annot_kws={"size": 14})
    plt.xlabel('Predicted labels', fontsize=14)
    plt.ylabel('True labels', fontsize=14)
    plt.title('Confusion Matrix', fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path, dpi=1200)
    plt.close()


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if not os.path.exists("./weights"):
        os.makedirs("./weights")

    tb_writer = SummaryWriter()

    train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(args.data_path)
    img_size = 224
    data_transform = {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(30),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406, 0.5], [0.229, 0.224, 0.225, 0.5])
        ]),
        "val": transforms.Compose([
            transforms.Resize(int(img_size * 1.143)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406, 0.5], [0.229, 0.224, 0.225, 0.5])
        ])
    }

    train_dataset = MyDataSet(images_path=train_images_path, images_class=train_images_label,
                              transform=data_transform["train"])
    val_dataset = MyDataSet(images_path=val_images_path, images_class=val_images_label, transform=data_transform["val"])

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    print('Using {} dataloader workers every process'.format(nw))

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True,
                                               num_workers=nw)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True,
                                             num_workers=nw)

    model = create_model(num_classes=args.num_classes).to(device)
    summary(model, (4, img_size, img_size))

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1E-4)

    validation_metrics = pd.DataFrame(columns=['Epoch', 'Validation Loss', 'Validation Accuracy'])
    roc_data = pd.DataFrame(columns=['Epoch', 'FPR', 'TPR', 'Thresholds'])  # 保存每轮的ROC曲线数据
    best_acc = 0.0
    best_cm = None
    best_y_true = []
    best_y_pred = []

    best_class_metrics = pd.DataFrame(columns=['Epoch', 'Class', 'AUC', 'ACC'])  # 存储每一轮每类AUC和ACC

    for epoch in range(args.epochs):
        train_one_epoch(model, optimizer, train_loader, device, epoch)
        val_loss, val_acc, y_true, y_pred = evaluate(model, val_loader, device, epoch)

        # ROC计算
        fpr, tpr, thresholds = roc_curve(y_true, y_pred, pos_label=1)
        roc_row = pd.DataFrame({
            'Epoch': [epoch + 1],
            'FPR': [fpr.tolist()],
            'TPR': [tpr.tolist()],
            'Thresholds': [thresholds.tolist()]
        })
        roc_data = pd.concat([roc_data, roc_row], ignore_index=True)

        new_row = pd.DataFrame({
            'Epoch': [epoch + 1],
            'Validation Loss': [val_loss],
            'Validation Accuracy': [val_acc]
        }, index=[0])
        validation_metrics = pd.concat([validation_metrics, new_row], ignore_index=True)

        # 保存每一类的AUC和ACC
        for class_idx in range(args.num_classes):
            y_true_binary = [1 if label == class_idx else 0 for label in y_true]
            y_pred_binary = [1 if label == class_idx else 0 for label in y_pred]

            try:
                class_auc = roc_auc_score(y_true_binary, y_pred_binary)
            except ValueError:
                class_auc = float('nan')  # 如果AUC无法计算（例如没有正类），用NaN表示

            class_acc = accuracy_score(y_true_binary, y_pred_binary)

            new_class_row = pd.DataFrame({
                'Epoch': [epoch + 1],
                'Class': [class_idx],
                'AUC': [class_auc],
                'ACC': [class_acc]
            })
            best_class_metrics = pd.concat([best_class_metrics, new_class_row], ignore_index=True)

        if val_acc > best_acc:
            best_acc = val_acc
            best_cm = confusion_matrix(y_true, y_pred)
            best_y_true = y_true
            best_y_pred = y_pred
            torch.save(model.state_dict(), "./weights/best_model.pth")

            # 绘制并保存最佳混淆矩阵
            print(f"Saving the best confusion matrix for epoch {epoch + 1}...")
            plot_confusion_matrix(best_cm, class_names=range(args.num_classes), output_path="best_confusion_matrix.png")

    # 保存指标数据
    validation_metrics.to_csv('validation_metrics.csv', index=False)
    best_class_metrics.to_csv('best_class_metrics.csv', index=False)
    roc_data.to_csv('roc_data.csv', index=False)  # 保存每轮ROC的横轴、纵轴数据
    torch.save(model.state_dict(), "./weights/latest_model.pth")

    best_model_path = "./weights/best_model.pth"
    if os.path.exists(best_model_path):
        print("Loading best model for t-SNE visualization...")
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        features, labels = extract_features(model, val_loader, device)
        plot_tsne(features, labels, colors, output_path="best_tsne_plot.png")
        print("Best t-SNE plot saved as best_tsne_plot.png.")

    # 绘制最后一轮的 t-SNE 图
    features, labels = extract_features(model, val_loader, device)
    plot_tsne(features, labels, colors, output_path="final_tsne_plot.png")
    print("Final t-SNE plot saved as final_tsne_plot.png.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=23)
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--data_path', type=str, default="D:/ConvNeXt/fiber")
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')
    args = parser.parse_args()
    main(args)
