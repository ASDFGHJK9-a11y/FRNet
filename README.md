
## Environment

The project is developed under the following environment:
- Python == 3.8
- PyTorch == 2.4.0
- CUDA == 11.8

For installation of the project dependencies, please run:

```sh
pip install -r requirements.txt
```

## Training

FRNet variants:

```sh
python train.py --savdir <path_to_save_weights> \
               --batch_size 8
               --epochs 80 \
               --lr 0.0002 \
               --image size 224\
               --data_dir <path_to_dataset>
```




