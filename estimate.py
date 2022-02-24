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

        if self.hparams.model['ema']:
            def avg_fn(averaged_model_parameter, model_parameter, _):
                α = self.hparams.model['momentum']
                return α * averaged_model_parameter + (1 - α) * model_parameter
            self.ema = AveragedModelWithBuffers(self.model, avg_fn=avg_fn)

        self.best_err = 1.0
        self.is_best_err = False

        self.real_T = torch.tensor(transition_matrix(
            self.hparams.model['backbone']['num_classes'],
            self.hparams['dataset']['noise_type'],
            self.hparams['dataset']['noise_ratio']))

    @property
    def phase_name(self):
        return 'trn' if self.training else 'val'

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        if self.hparams.model['ema']:
            self.ema.update_parameters(self.model)

    def forward(self, x):
        if self.training or not self.hparams.model['ema']:
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
        err = 1 - self.accuracy[phase].compute()
        self.log(f'{phase}/err', err)
        self.accuracy[phase].reset()
        return err

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
        err = self.shared_epoch_end(outputs[0])
        self.is_best_err = (err < self.best_err)
        if self.is_best_err:
            self.best_err = err
        self.test_epoch_end(outputs[1])

    def test_step(self, batch, batch_idx):
        x, y = batch
        ỹ = self(x).softmax(dim=-1)
        return {'real': y, 'pred': ỹ}

    def test_epoch_end(self, outputs):
        C = self.hparams.model['backbone']['num_classes']
        E = self.current_epoch
        real = torch.concat([x['real'].cpu() for x in outputs])
        pred = torch.concat([x['pred'].cpu() for x in outputs])
        conf = pred.max(dim=1).values
        info = -pred.xlogy(pred).sum(dim=1)

        if E == 0:
            self.avg_conf = conf
        else:
            self.avg_conf = (self.avg_conf * E + conf) / (E + 1)

        if E == 0:
            self.avg_info = info
        else:
            self.avg_info = (self.avg_info * E + info) / (E + 1)

        # just average
        T1 = torch.zeros((C, C))
        for y, ỹ in zip(real, pred):
            T1[y] += ỹ
        T1 /= T1.sum(axis=1, keepdim=True)

        # max confidence
        T2 = torch.zeros((C, C))
        max_conf = torch.zeros(C)
        for y, ỹ, c in zip(real, pred, conf):
            if max_conf[y] < c:
                max_conf[y] = c
                T2[y] = ỹ

        # min entropy
        T3 = torch.zeros((C, C))
        min_info = torch.ones(C) * torch.tensor(C).log()
        for y, ỹ, h in zip(real, pred, info):
            if min_info[y] > h:
                min_info[y] = h
                T3[y] = ỹ

        # max average confidence
        T4 = torch.zeros((C, C))
        max_avg_conf = torch.zeros(C)
        for y, ỹ, c in zip(real, pred, self.avg_conf):
            if max_avg_conf[y] < c:
                max_avg_conf[y] = c
                T4[y] = ỹ

        # min average entropy
        T5 = torch.zeros((C, C))
        min_avg_info = torch.ones(C) * torch.tensor(C).log()
        for y, ỹ, h in zip(real, pred, self.avg_info):
            if min_avg_info[y] > h:
                min_avg_info[y] = h
                T5[y] = ỹ

        # average weighted by average confidence
        T6 = torch.zeros((C, C))
        for y, ỹ, c in zip(real, pred, self.avg_conf):
            T6[y] += ỹ * c
        T6 /= T6.sum(axis=1, keepdim=True)

        # average weighted by e^(-average entropy)
        T7 = torch.zeros((C, C))
        for y, ỹ, h in zip(real, pred, self.avg_info):
            T7[y] += ỹ * (-h).exp()
        T7 /= T7.sum(axis=1, keepdim=True)

        self.logger.experiment.add_image('T1', T1, self.current_epoch, dataformats='HW')
        self.logger.experiment.add_image('T2', T2, self.current_epoch, dataformats='HW')
        self.logger.experiment.add_image('T3', T3, self.current_epoch, dataformats='HW')
        self.logger.experiment.add_image('T4', T4, self.current_epoch, dataformats='HW')
        self.logger.experiment.add_image('T5', T5, self.current_epoch, dataformats='HW')
        self.logger.experiment.add_image('T6', T6, self.current_epoch, dataformats='HW')
        self.logger.experiment.add_image('T7', T7, self.current_epoch, dataformats='HW')
        # self.logger.experiment.add_figure(
        #     'T1', plot_confusion_matrix(T1), self.current_epoch)
        # self.logger.experiment.add_figure(
        #     'T2', plot_confusion_matrix(T2), self.current_epoch)
        # self.logger.experiment.add_figure(
        #     'T3', plot_confusion_matrix(T3), self.current_epoch)
        # self.logger.experiment.add_figure(
        #     'T4', plot_confusion_matrix(T4), self.current_epoch)
        # self.logger.experiment.add_figure(
        #     'T5', plot_confusion_matrix(T5), self.current_epoch)
        # self.logger.experiment.add_figure(
        #     'T6', plot_confusion_matrix(T6), self.current_epoch)
        # self.logger.experiment.add_figure(
        #     'T7', plot_confusion_matrix(T7), self.current_epoch)

        if self.is_best_err:
            save_path = os.path.join(self.logger.log_dir, 'T')
            if not os.path.isdir(save_path):
                os.makedirs(save_path)
            torch.save(T1, os.path.join(save_path, f'T1.pth'))
            torch.save(T2, os.path.join(save_path, f'T2.pth'))
            torch.save(T3, os.path.join(save_path, f'T3.pth'))
            torch.save(T4, os.path.join(save_path, f'T4.pth'))
            torch.save(T5, os.path.join(save_path, f'T5.pth'))
            torch.save(T6, os.path.join(save_path, f'T6.pth'))
            torch.save(T6, os.path.join(save_path, f'T7.pth'))

        self.log('val/rmse/T1', ((self.real_T - T1) ** 2).mean().sqrt())
        self.log('val/rmse/T2', ((self.real_T - T2) ** 2).mean().sqrt())
        self.log('val/rmse/T3', ((self.real_T - T3) ** 2).mean().sqrt())
        self.log('val/rmse/T4', ((self.real_T - T4) ** 2).mean().sqrt())
        self.log('val/rmse/T5', ((self.real_T - T5) ** 2).mean().sqrt())
        self.log('val/rmse/T6', ((self.real_T - T6) ** 2).mean().sqrt())
        self.log('val/rmse/T7', ((self.real_T - T7) ** 2).mean().sqrt())

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

    noisy_data = dm.datasets['noisy']
    noisy_train, noisy_val = torch.utils.data.random_split(
        noisy_data,
        [len(noisy_data) - len(noisy_data) // 10, len(noisy_data) // 10])
    train_loader = DataLoader(
        noisy_train,
        batch_size=hparams['dataset']['batch_sizes']['train'],
        shuffle=True,
        num_workers=os.cpu_count(),
        pin_memory=True
    )
    val_loader = DataLoader(
        noisy_val,
        batch_size=hparams['dataset']['batch_sizes']['val'],
        num_workers=os.cpu_count(),
        pin_memory=True
    )
    clean_loader = DataLoader(
        dm.datasets['clean'].datasets[0],
        batch_size=hparams['dataset']['batch_sizes']['val'],
        num_workers=os.cpu_count(), pin_memory=True)

    trainer = pl.Trainer.from_argparse_args(
        args,
        logger=TensorBoardLogger(
            f"lightning_logs/matrix/{hparams['dataset']['name']}",
            f"{args.num_clean}-{args.noise_type}-{args.noise_ratio}-{args.random_seed}"),
        callbacks=[
            ModelCheckpoint(monitor='val/err', save_top_k=1, save_last=True),
            LearningRateMonitor(),
        ]
    )

    pl_module = NoiseEstimator(**hparams, steps_per_epoch=len(train_loader))

    trainer.fit(pl_module, train_loader, [val_loader, clean_loader])


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
