import os
import json
import argparse

import torch
from pytorch_lightning import Trainer

from methods import NoisyFlexMatchClassifier
from weaver.transforms import get_trfms
from synthetic_noisy_datasets import NoisyCIFAR10, NoisyCIFAR100

DataModule = {
    'cifar10': NoisyCIFAR10,
    'cifar100': NoisyCIFAR100,
}


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

    test(config, args)
