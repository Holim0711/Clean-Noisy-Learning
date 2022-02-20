import os
import json
import argparse

import numpy as np
import matplotlib.pyplot as plt
from itertools import product

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.utilities.seed import seed_everything
from torchmetrics import Accuracy

from weaver.models import get_model
from weaver.optimizers import get_optim
from weaver.schedulers import get_sched
from weaver.transforms import get_trfms

from deficient_cifar import NoisyCIFAR10, NoisyCIFAR100
from deficient_cifar.utils import *

DataModule = {
    'cifar10': NoisyCIFAR10,
    'cifar100': NoisyCIFAR100,
}


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


class AveragedModelWithBuffers(torch.optim.swa_utils.AveragedModel):
    def update_parameters(self, model):
        super().update_parameters(model)
        for a, b in zip(self.module.buffers(), model.buffers()):
            a.copy_(b.to(a.device))


def change_bn(model, momentum):
    if isinstance(model, nn.BatchNorm2d):
        model.momentum = 1 - momentum
    else:
        for children in model.children():
            change_bn(children, momentum)


class NoiseEstimator(pl.LightningModule):

    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.model = get_model(**self.hparams.model['backbone'])
        change_bn(self.model, self.hparams.model['momentum'])
        self.criterion = nn.CrossEntropyLoss()
        self.accuracy = nn.ModuleDict({k: Accuracy() for k in ['trn', 'val']})

        def avg_fn(averaged_model_parameter, model_parameter, num_averaged):
            α = self.hparams.model['momentum']
            return α * averaged_model_parameter + (1 - α) * model_parameter
        self.ema = AveragedModelWithBuffers(self.model, avg_fn=avg_fn)

    @property
    def phase_name(self):
        return 'trn' if self.training else 'val'

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        self.ema.update_parameters(self.model)

    def forward(self, x):
        if self.training:
            return self.model(x)
        else:
            return self.ema(x)

    def shared_step(self, batch):
        x, y = batch
        z = self(x)
        loss = self.criterion(z, y)
        self.log(f'{self.phase_name}/loss', loss)
        self.accuracy[self.phase_name].update(z.softmax(dim=-1), y)
        return loss

    def shared_epoch_end(self, outputs):
        phase = self.phase_name
        self.log(f'{phase}/err', 1 - self.accuracy[phase].compute())
        self.accuracy[phase].reset()

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch)

    def training_epoch_end(self, outputs):
        self.shared_epoch_end(outputs)

    def validation_step(self, batch, batch_idx, dataloader_idx):
        if dataloader_idx == 0:
            return self.shared_step(batch)
        elif dataloader_idx == 1:
            return self.test_step(batch, batch_idx)

    def validation_epoch_end(self, outputs):
        self.shared_epoch_end(outputs[0])
        self.test_epoch_end(outputs[1])

    def test_step(self, batch, batch_idx):
        x, y = batch
        ỹ = self.ema(x).softmax(dim=-1)
        return {'real': y, 'pred': ỹ}

    def test_epoch_end(self, outputs):
        C = self.hparams.model['backbone']['num_classes']
        max_conf = torch.zeros(C)
        pred_T = torch.zeros((C, C))

        for x in outputs:
            for y, ỹ in zip(x['real'].cpu(), x['pred'].cpu()):
                if max_conf[y] < ỹ.max():
                    max_conf[y] = ỹ.max()
                    pred_T[y] = ỹ

        torch.save(pred_T, os.path.join(self.logger.log_dir, 'T.pth'))
        fig = plot_confusion_matrix(pred_T)
        self.logger.experiment.add_figure('matrix', fig, self.current_epoch)

        real_T = torch.tensor(transition_matrix(
            self.hparams.model['backbone']['num_classes'],
            self.hparams['dataset']['noise_type'],
            self.hparams['dataset']['noise_ratio']))

        err = real_T - pred_T
        self.log('val/rmse', (err * err).mean().sqrt())

    def configure_optimizers(self):
        optim = get_optim(self, **self.hparams.optimizer)
        sched = get_sched(optim, **self.hparams.scheduler)
        sched.extend(self.hparams.steps_per_epoch)
        return {'optimizer': optim,
                'lr_scheduler': {'scheduler': sched, 'interval': 'step'}}


def train(dm, hparams, args):
    hparams['dataset']['num_clean'] = args.num_clean
    hparams['dataset']['noise_type'] = args.noise_type
    hparams['dataset']['noise_ratio'] = args.noise_ratio
    hparams['dataset']['random_seed'] = args.random_seed

    dm.prepare_data()
    dm.setup()

    train_loader = dm.dataloader('noisy')
    val_loader = dm.val_dataloader()
    test_loader = DataLoader(
        dm.datasets['clean'].datasets[0],
        batch_size=hparams['dataset']['batch_sizes']['val'],
        num_workers=os.cpu_count(), pin_memory=True)

    trainer = pl.Trainer.from_argparse_args(
        args,
        logger=TensorBoardLogger(
            f"lightning_logs/matrix/{hparams['dataset']['name']}",
            f"{args.num_clean}-{args.noise_type}-{args.noise_ratio}-{args.random_seed}"),
        callbacks=[
            ModelCheckpoint(save_last=True),
            LearningRateMonitor(),
        ]
    )

    pl_module = NoiseEstimator(**hparams, steps_per_epoch=len(train_loader))

    trainer.fit(pl_module, train_loader, [val_loader, test_loader])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    parser.add_argument('--num_clean', type=int, default=100)
    parser.add_argument('--noise_type', type=str, default='symmetric')
    parser.add_argument('--noise_ratio', type=float, default=0.2)
    parser.add_argument('--random_seed', type=int, default=0)
    parser = pl.Trainer.add_argparse_args(parser)

    args = parser.parse_args()
    seed_everything(args.random_seed)

    with open(args.config) as file:
        hparams = json.load(file)

    dm = DataModule[hparams['dataset']['name']](
        root=os.path.join('data', hparams['dataset']['name']),
        num_clean=args.num_clean,
        noise_type=args.noise_type,
        noise_ratio=args.noise_ratio,
        transforms={
            'noisy': get_trfms(hparams['transforms']['train']),
            'val': get_trfms(hparams['transforms']['val']),
        },
        batch_sizes={
            'clean': 0,
            'noisy': hparams['dataset']['batch_sizes']['train'],
            'val': hparams['dataset']['batch_sizes']['val'],
        },
        random_seed=args.random_seed,
        pure_unproved=True,
    )

    train(dm, hparams, args)
