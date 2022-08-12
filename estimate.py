import os
import json
import argparse

import numpy as np
import matplotlib.pyplot as plt
from itertools import product

import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.utilities.seed import seed_everything
from torchmetrics import Accuracy

from weaver.models import get_classifier
from weaver.optimizers import get_optim
from weaver.schedulers import get_sched
from weaver.transforms import get_xform

from synthetic_noisy_datasets import NoisyCIFAR10, NoisyCIFAR100


def plot_confusion_matrix(cm, labs=None, ylabs=None, xlabs=None, cmap='Blues'):
    """
    Reference: scikit-learn/sklearn/metrics/_plot/confusion_matrix.py
    Args:
        - cm: confusion matrix (N x M)
        - lab: label texts (only when N == M)
        - ylab: true label texts
        - xlab: pred label texts
        - cmap: color theme (see 'colormaps' in matplotlib)
    """
    fig, ax = plt.subplots()

    im_ = ax.imshow(cm, interpolation='nearest', cmap=cmap, vmin=0., vmax=1.)
    cmap_min, cmap_max = im_.cmap(0), im_.cmap(256)

    for i, j in product(range(cm.shape[0]), range(cm.shape[1])):
        color = cmap_max if cm[i, j] < 0.5 else cmap_min
        text = f'{cm[i, j]:.2f}'
        text = '1.0' if text[0] == '1' else text[1:]
        ax.text(j, i, text, ha="center", va="center", color=color)

    if ylabs is None:
        ylabs = np.arange(cm.shape[0]) if labs is None else labs

    if xlabs is None:
        xlabs = np.arange(cm.shape[1]) if labs is None else labs

    fig.colorbar(im_, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=xlabs,
           yticklabels=ylabs,
           ylabel="True label",
           xlabel="Predicted label")

    ax.set_ylim((cm.shape[0] - 0.5, -0.5))
    plt.setp(ax.get_xticklabels(), rotation='vertical')

    return fig


class NoiseEstimator(pl.LightningModule):

    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.model = get_classifier(**self.hparams.model['backbone'])
        self.criterion = nn.CrossEntropyLoss()
        self.accuracy = nn.ModuleDict({k: Accuracy() for k in ['trn', 'val']})

    @property
    def phase_name(self):
        return 'trn' if self.training else 'val'

    def forward(self, x):
        return self.model(x)

    def shared_step(self, batch):
        x, y = batch
        z = self(x)
        loss = self.criterion(z, y)
        self.log(f'{self.phase_name}/loss', loss)
        self.accuracy[self.phase_name].update(z.softmax(dim=-1), y)
        return loss

    def shared_epoch_end(self, outputs):
        phase = self.phase_name
        acc = self.accuracy[phase].compute()
        self.log(f'{phase}/acc', acc)
        self.accuracy[phase].reset()

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch)

    def training_epoch_end(self, outputs):
        self.shared_epoch_end(outputs)

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch)

    def validation_epoch_end(self, outputs):
        self.shared_epoch_end(outputs[0])

    def configure_optimizers(self):
        optim = get_optim(self, **self.hparams.optimizer)
        sched = get_sched(optim, **self.hparams.scheduler)
        return {'optimizer': optim, 'lr_scheduler': {'scheduler': sched}}


def train(args):
    config = args.config
    seed_everything(config['random_seed'])

    trainer = pl.Trainer.from_argparse_args(
        args,
        logger=TensorBoardLogger('logs/estimate', config['dataset']['name']),
        callbacks=[
            ModelCheckpoint(save_last=True,
                            save_top_k=1, monitor='val/acc', mode='max'),
            LearningRateMonitor(),
        ]
    )

    xform_train = get_xform('Compose', transforms=config['transform']['train'])
    xform_val = get_xform('Compose', transforms=config['transform']['val'])

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
            'noisy': xform_train,
            'val': xform_val,
        },
        batch_sizes={
            'clean': 0,
            'noisy': config['dataset']['batch_sizes']['train'],
            'val': config['dataset']['batch_sizes']['val'],
        },
        random_seed=config['dataset']['random_seed'],
    )

    pl_module = NoiseEstimator(**config)

    trainer.fit(pl_module, dm)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=lambda x: json.load(open(x)))
    parser.add_argument('--dataset.num_clean', type=int)
    parser.add_argument('--dataset.noise_type', type=str)
    parser.add_argument('--dataset.noise_ratio', type=float)
    parser.add_argument('--dataset.random_seed', type=int)
    parser.add_argument('--random_seed', type=int)
    parser = pl.Trainer.add_argparse_args(parser)

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
