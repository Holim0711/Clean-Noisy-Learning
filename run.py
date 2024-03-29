import os
import json
import argparse

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.utilities.seed import seed_everything

from methods import NoisyFlexMatchClassifier
from weaver.transforms import get_trfms
from weaver.transforms.twin_transforms import NqTwinTransform
from deficient_cifar import NoisyCIFAR10, NoisyCIFAR100

DataModule = {
    'cifar10': NoisyCIFAR10,
    'cifar100': NoisyCIFAR100,
}


def train(config, args):
    seed_everything(args.random_seed)
    config['dataset']['num_clean'] = args.num_clean
    config['dataset']['noise_type'] = args.noise_type
    config['dataset']['noise_ratio'] = args.noise_ratio
    config['dataset']['random_seed'] = args.random_seed

    trainer = Trainer.from_argparse_args(
        args,
        logger=TensorBoardLogger('logs', config['dataset']['name']),
        callbacks=[
            ModelCheckpoint(save_top_k=1, monitor='val/acc', mode='max'),
            LearningRateMonitor(),
        ]
    )

    n_device = trainer.num_nodes * (trainer.num_gpus or trainer.num_processes)
    assert config['dataset']['batch_sizes']['clean'] % n_device == 0
    assert config['dataset']['batch_sizes']['noisy'] % n_device == 0

    transform_w = get_trfms(config['transform']['weak'])
    transform_s = get_trfms(config['transform']['strong'])
    transform_v = get_trfms(config['transform']['val'])

    dm = DataModule[config['dataset']['name']](
        root=os.path.join('data', config['dataset']['name']),
        num_clean=args.num_clean,
        noise_type=args.noise_type,
        noise_ratio=args.noise_ratio,
        transforms={
            'clean': transform_w,
            'noisy': NqTwinTransform(transform_s, transform_w),
            'val': transform_v
        },
        batch_sizes={
            'clean': config['dataset']['batch_sizes']['clean'] // n_device,
            'noisy': config['dataset']['batch_sizes']['noisy'] // n_device,
            'val': config['dataset']['batch_sizes']['val'],
        },
        random_seed=args.random_seed,
        enum_unproved=True,
    )
    dm.prepare_data()
    dm.setup()

    ỹ = dm.datasets['noisy'].dataset.targets
    T = dm.T if args.T is None else torch.load(args.T)

    model = NoisyFlexMatchClassifier(ỹ, T, **config)

    trainer.fit(model, dm.train_dataloader(), dm.val_dataloader())


def test(config, args):
    trainer = Trainer.from_argparse_args(args, logger=False)
    dm = DataModule[config['dataset']['name']](
        root=os.path.join('data', config['dataset']['name']),
        num_clean=0,
        transforms={'val': get_trfms(config['transform']['val'])},
        batch_sizes={'val': config['dataset']['batch_sizes']['val']},
    )
    dm.prepare_data()
    dm.setup()
    ỹ = dm.datasets['noisy'].targets
    T = dm.T
    model = NoisyFlexMatchClassifier.load_from_checkpoint(args.ckpt_path, ỹ=ỹ, T=T)
    trainer.test(model, dm.test_dataloader())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['train', 'test'])
    parser.add_argument('config', type=str)
    parser.add_argument('--num_clean', type=int, default=100)
    parser.add_argument('--noise_type', type=str, default='symmetric')
    parser.add_argument('--noise_ratio', type=float, default=0.2)
    parser.add_argument('--random_seed', type=int, default=0)
    parser.add_argument('--T', type=str)
    parser.add_argument('--ckpt_path', type=str)
    parser = Trainer.add_argparse_args(parser)

    args = parser.parse_args()

    with open(args.config) as file:
        config = json.load(file)

    if args.mode == 'train':
        train(config, args)
    else:
        test(config, args)
