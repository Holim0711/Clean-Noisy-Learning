import os
import json
import argparse

from torchvision.datasets import CIFAR10, CIFAR100
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer

from methods import NoisyFlexMatchClassifier
from weaver.transforms import get_xform


def test(args):
    config = args.config

    trainer = Trainer.from_argparse_args(args, logger=False)

    Dataset = {
        'CIFAR10': CIFAR10,
        'CIFAR100': CIFAR100,
    }[config['dataset']['name']]

    dataset = Dataset(
        root=os.path.join('data', config['dataset']['name']),
        train=False,
        transform=get_xform('Compose', transforms=config['transform']['val'])
    )

    dataloader = DataLoader(
        dataset,
        batch_size=config['dataset']['batch_sizes']['val'],
        shuffle=False,
        num_workers=os.cpu_count(),
        pin_memory=True
    )

    model = NoisyFlexMatchClassifier.load_from_checkpoint(args.checkpoint)
    trainer.test(model, dataloader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=lambda x: json.load(open(x)))
    parser.add_argument('checkpoint', type=str)
    parser = Trainer.add_argparse_args(parser)

    args = parser.parse_args()
    test(args)
