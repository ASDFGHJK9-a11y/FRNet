#Usage

## Requirements

- Python >= 3.8
- PyTorch >= 2.4.0
- CUDA >= 11.8
- NumPy >= 1.24.3

## Training

To train the model, use the following command:

```sh
python main.py --savdir <path_to_save_weights> \
               --batch_size 8 \
               --epochs 80 \
               --lr 0.0002 \
               --data_dir <path_to_dataset>
```




