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
    config['dataset']['n'] = args.n
    config['dataset']['noise_type'] = args.noise_type
    config['dataset']['noise_ratio'] = args.noise_ratio
    config['dataset']['random_seed'] = args.random_seed
    seed_everything(args.random_seed)

    trainer = Trainer.from_argparse_args(
        args,
        logger=TensorBoardLogger('logs', config['dataset']['name']),
        callbacks=[
            ModelCheckpoint(save_top_k=1, monitor='val/acc', mode='max'),
            LearningRateMonitor(),
        ]
    )

    nd = trainer.num_nodes * trainer.num_gpus

    transform_w = get_trfms(config['transform']['weak'])
    transform_s = get_trfms(config['transform']['strong'])
    transform_v = get_trfms(config['transform']['val'])

    dm = DataModule[config['dataset']['name']](
        os.path.join('data', config['dataset']['name']),
        args.n,
        noise_type=args.noise_type,
        noise_ratio=args.noise_ratio,
        transforms={
            'clean': transform_w,
            'noisy': NqTwinTransform(transform_s, transform_w),
            'val': transform_v
        },
        batch_sizes={
            'clean': config['dataset']['batch_sizes']['clean'] // nd,
            'noisy': config['dataset']['batch_sizes']['noisy'] // nd,
            'val': config['dataset']['batch_sizes']['val'],
        },
        random_seed=args.random_seed,
        show_sample_indices=True,
    )
    dm.prepare_data()
    dm.setup()
    ỹ = [x[0] for x in dm.datasets['noisy'].dataset.targets]
    T = dm.T    # T = torch.load(args.transition_matrix)

    model = NoisyFlexMatchClassifier(ỹ, T, **config)
    trainer.fit(model, dm)


def test(config, args):
    trainer = Trainer.from_argparse_args(args, logger=False)
    dm = DataModule[config['dataset']['name']](
        os.path.join('data', config['dataset']['name']),
        n=0,
        transforms={'val': get_trfms(config['transform']['val'])},
        batch_sizes={'val': config['dataset']['batch_sizes']['val']},
    )
    model = NoisyFlexMatchClassifier.load_from_checkpoint(args.ckpt_path)
    trainer.test(model, dm)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['train', 'test'])
    parser.add_argument('config', type=str)
    parser.add_argument('--n', type=int, default=40)
    parser.add_argument('--noise_type', type=str, default='symmetric')
    parser.add_argument('--noise_ratio', type=float, default=0.2)
    parser.add_argument('--random_seed', type=int, default=1234)
    # parser.add_argument('--transition_matrix', type=str)
    parser.add_argument('--ckpt_path', type=str)
    parser = Trainer.add_argparse_args(parser)

    args = parser.parse_args()

    with open(args.config) as file:
        config = json.load(file)

    if args.mode == 'train':
        train(config, args)
    else:
        test(config, args)


if __name__ == "__main__":
    main()
