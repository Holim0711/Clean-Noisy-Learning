import os
import json
import argparse
from math import ceil

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.utilities.argparse import parse_env_variables
from lightning_lite import seed_everything
import torch
from torch.utils.data import DataLoader, ConcatDataset
from torchvision.datasets import CIFAR10, CIFAR100
from torchvision.transforms import Compose, Lambda
from weaver import get_transforms
from weaver.datasets import RandomSubset, IndexedDataset
from noisy_cifar import NoisyCIFAR10, NoisyCIFAR100

from methods import NoisyFlexMatchClassifier


def train(hparams):
    seed_everything(hparams.get('random_seed', 0))

    trainer = Trainer(
        **vars(parse_env_variables(Trainer)),
        logger=TensorBoardLogger('logs', hparams['dataset']['name']),
        callbacks=[
            ModelCheckpoint(save_top_k=1, monitor='val/acc/ema', mode='max'),
            LearningRateMonitor(),
        ]
    )

    transform_w = Compose(get_transforms(hparams['transform']['weak']))
    transform_s = Compose(get_transforms(hparams['transform']['strong']))
    transform_v = Compose(get_transforms(hparams['transform']['val']))

    CleanDataset, NoisyDataset = {
        'CIFAR10': (CIFAR10, NoisyCIFAR10),
        'CIFAR100': (CIFAR100, NoisyCIFAR100)
    }[hparams['dataset']['name']]

    noisy_dataset = NoisyDataset(
        hparams['dataset']['root'],
        hparams['dataset']['noise_type'],
        hparams['dataset']['noise_level'],
        hparams['dataset']['random_seed'],
        transform=Lambda(lambda x: (transform_s(x), transform_w(x))),
    )
    noisy_dataset = IndexedDataset(noisy_dataset)

    clean_dataset = CleanDataset(
        hparams['dataset']['root'],
        transform=transform_w,
    )
    clean_dataset = RandomSubset(
        clean_dataset,
        hparams['dataset']['num_clean'],
        class_balanced=True,
        random_seed=hparams['dataset']['random_seed'],
    )
    batch_size = hparams['dataset']['batch_size']
    noisy_num_iters = len(noisy_dataset) / batch_size['noisy']
    clean_num_iters = len(clean_dataset) / batch_size['clean']
    m = ceil(noisy_num_iters / (clean_num_iters * 2))
    clean_dataset = ConcatDataset([clean_dataset] * m)

    val_dataset = CleanDataset(
        hparams['dataset']['root'],
        train=False,
        transform=transform_v,
    )

    noisy_dataloader = DataLoader(
        noisy_dataset, batch_size['noisy'],
        shuffle=True, num_workers=os.cpu_count(), pin_memory=True)
    clean_dataloader = DataLoader(
        clean_dataset, batch_size['clean'],
        shuffle=False, num_workers=os.cpu_count(), pin_memory=True)
    train_dataloader = {
        'clean': clean_dataloader,
        'noisy': noisy_dataloader,
    }
    val_dataloader = DataLoader(
        val_dataset, batch_size['clean'] + batch_size['noisy'],
        shuffle=False, num_workers=os.cpu_count(), pin_memory=True)

    if hparams['method'] == 'noisy-flexmatch':
        model = NoisyFlexMatchClassifier(**hparams)
        model.criterionᵤ.initialize_constants(
            torch.load(os.path.join(
                'transition_matrix',
                hparams['dataset']['name'],
                hparams['dataset']['noise_type']
                + f"-{hparams['dataset']['noise_level']}"
                + f"-{hparams['dataset']['random_seed']}.pth"
            ))['inv_cur_h'],
            noisy_dataset.dataset.targets
        )

    trainer.fit(model, train_dataloader, val_dataloader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=lambda x: json.load(open(x)))
    parser.add_argument('--dataset.noise_type',
                        choices=['symmetric', 'asymmetric', 'human'])
    parser.add_argument('--dataset.noise_level',
                        type=lambda x: float(x) if x.startswith('0.') else x)
    parser.add_argument('--dataset.random_seed', type=int)

    args = parser.parse_args()
    if (v := getattr(args, 'dataset.noise_type')) is not None:
        args.config['dataset']['noise_type'] = v
    if (v := getattr(args, 'dataset.noise_level')) is not None:
        args.config['dataset']['noise_level'] = v
    if (v := getattr(args, 'dataset.random_seed')) is not None:
        args.config['dataset']['random_seed'] = v

    train(args.config)
