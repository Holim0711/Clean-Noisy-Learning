import os
import json
import argparse

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.utilities.seed import seed_everything

from weaver.transforms import get_xform
from weaver.transforms.twin_transforms import NqTwinTransform
from synthetic_noisy_datasets.datamodules import NoisyCIFAR10, NoisyCIFAR100

from methods import NoisyFlexMatchClassifier


def train(args):
    config = args.config
    seed_everything(config['random_seed'])

    trainer = Trainer.from_argparse_args(
        args,
        logger=TensorBoardLogger('logs', config['dataset']['name']),
        callbacks=[
            ModelCheckpoint(save_top_k=1, monitor='val/acc/ema', mode='max'),
            LearningRateMonitor(),
        ]
    )

    N = trainer.num_nodes * trainer.num_devices

    transform_w = get_xform('Compose', transforms=config['transform']['weak'])
    transform_s = get_xform('Compose', transforms=config['transform']['str'])
    transform_v = get_xform('Compose', transforms=config['transform']['val'])

    Dataset = {
        'CIFAR10': NoisyCIFAR10,
        'CIFAR100': NoisyCIFAR100,
    }[config['dataset']['name']]

    dm = Dataset(
        os.path.join('data', config['dataset']['name']),
        config['dataset']['num_clean'],
        config['dataset']['noise_type'],
        config['dataset']['noise_ratio'],
        transforms={
            'clean': transform_w,
            'noisy': NqTwinTransform(transform_s, transform_w),
            'val': transform_v
        },
        batch_sizes={
            'clean': config['dataset']['batch_sizes']['clean'] // N,
            'noisy': config['dataset']['batch_sizes']['noisy'] // N,
            'val': config['dataset']['batch_sizes']['val'],
        },
        random_seed=config['dataset']['random_seed'],
        expand_clean=True,
        enumerate_noisy=True,
    )

    if config['method'] == 'noisy-flexmatch':
        T = torch.load(args.T)
        model = NoisyFlexMatchClassifier(T=args.T, **config)

    trainer.fit(model, dm)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=lambda x: json.load(open(x)))
    parser.add_argument('--dataset.num_clean', type=int)
    parser.add_argument('--dataset.noise_type', type=str)
    parser.add_argument('--dataset.noise_ratio', type=float)
    parser.add_argument('--dataset.random_seed', type=int)
    parser.add_argument('--random_seed', type=int)
    parser.add_argument('--T', type=str)
    parser = Trainer.add_argparse_args(parser)

    args = parser.parse_args()
    if (v := getattr(args, 'dataset.num_clean')) is not None:
        args.config['dataset']['num_clean'] = v
    if (v := getattr(args, 'dataset.noise_type')) is not None:
        args.config['dataset']['noise_type'] = v
    if (v := getattr(args, 'dataset.noise_ratio')) is not None:
        args.config['dataset']['noise_ratio'] = v
    if (v := getattr(args, 'dataset.random_seed')) is not None:
        args.config['dataset']['random_seed'] = v
    if (v := getattr(args, 'random_seed')) is not None:
        args.config['random_seed'] = v
    train(args)
