# NoisyFlexMatch

## Dependencies
- PyTorch
- PyTorch-Lightning
- Weaver (https://github.com/Holim0711/Weaver)
- Synthetic-Noisy-Datasets (https://github.com/Holim0711/Synthetic-Noisy-Datasets)

## Training
```
python train.py configs/cifar10.json --gpus 1 --max_epoch 9362 --random_seed 0 --deterministic
```
