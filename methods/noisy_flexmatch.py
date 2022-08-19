import os
import numpy as np
import torch
import pytorch_lightning as pl
from torchmetrics import Accuracy
from weaver.models import get_classifier
from weaver.optimizers import get_optim
from weaver.optimizers.utils import exclude_wd
from weaver.schedulers import get_sched
from .utils import EMA, change_bn_momentum, replace_relu_to_lrelu

__all__ = ['NoisyFlexMatchClassifier']


class NoisyFlexMatchCrossEntropy(torch.nn.Module):

    def __init__(self, num_classes, num_samples, temperature, threshold):
        super().__init__()
        self.threshold = threshold
        self.temperature = temperature
        self.𝜇ₘₐₛₖ = None

        self.num_classes = num_classes
        self.num_samples = num_samples

        self.T = torch.eye(num_classes)
        self.Ỹ = torch.tensor([num_classes] * num_samples)
        self.Ŷ = torch.tensor([num_classes] * num_samples)

        self.ones = torch.ones(num_samples)
        # self.Dỹ = self.Ỹ.bincount().view(-1, 1) / self.num_samples

    def forward(self, logits_s, logits_w, ỹ):
        Tŷỹ = torch.zeros((self.num_classes + 1, self.num_classes))
        Tŷỹ.index_put_((self.Ŷ, self.Ỹ), self.ones, True)
        Tŷỹ = Tŷỹ[:-1] + 1  # try?: (self.Dỹ * Tŷỹ[-1])
        Tŷỹ = Tŷỹ / Tŷỹ.sum(axis=1, keepdims=True)

        α = self.T / Tŷỹ
        α = α.t().to(ỹ.device)

        probs = torch.softmax(logits_w / self.temperature, dim=-1)
        probs *= α[ỹ]
        probs /= probs.sum(dim=-1, keepdim=True)
        max_probs, targets = probs.max(dim=-1)

        β = self.Ŷ.bincount()
        β = β / β.max()
        β = β / (2 - β)
        β = β.to(targets.device)
        masks = (max_probs > self.threshold * β[targets]).float()

        self.ŷ = torch.where(max_probs > self.threshold, targets, -1)

        loss = torch.nn.functional.cross_entropy(
            logits_s, targets, reduction='none') * masks
        self.𝜇ₘₐₛₖ = masks.float().mean().detach()

        return loss.mean()


class NoisyFlexMatchClassifier(pl.LightningModule):

    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.model = get_classifier(**self.hparams.model['backbone'])
        if a := self.hparams.model.get('lrelu'):
            replace_relu_to_lrelu(self.model, a)
        self.criterionₗ = torch.nn.CrossEntropyLoss()
        self.criterionᵤ = NoisyFlexMatchCrossEntropy(
            self.hparams.model['backbone']['num_classes'],
            {
                'CIFAR10': 50000,
                'CIFAR100': 50000,
            }[self.hparams.dataset['name']],
            **self.hparams.model['loss_u']
        )
        self.train_acc = Accuracy()
        self.val_acc = Accuracy()

        if m := self.hparams.model.get('ema'):
            change_bn_momentum(self.model, m)
            self.ema = EMA(self.model, m)
            self.val_acc_ema = Accuracy()

    def on_train_start(self):
        self.criterionᵤ.T = torch.load(self.hparams.T)
        self.criterionᵤ.Ỹ = torch.from_numpy(np.load(os.path.join(
            'data', self.hparams.dataset['name'], 'noisy') +
            f"-{self.hparams.dataset['noise_type']}" +
            f"-{self.hparams.dataset['noise_ratio']}" +
            f"-{self.hparams.dataset['random_seed']}.npy"))

    def training_step(self, batch, batch_idx):
        xₗ, yₗ = batch['clean']
        iᵤ, ((ˢxᵤ, ʷxᵤ), ỹ) = batch['noisy']

        z = self.model(torch.cat((xₗ, ˢxᵤ, ʷxᵤ)))
        zₗ = z[:xₗ.shape[0]]
        ˢzᵤ, ʷzᵤ = z[xₗ.shape[0]:].chunk(2)
        del z

        lossₗ = self.criterionₗ(zₗ, yₗ)
        lossᵤ = self.criterionᵤ(ˢzᵤ, ʷzᵤ.detach(), ỹ)
        loss = lossₗ + lossᵤ

        ŷᵤ = self.criterionᵤ.ŷ
        if torch.distributed.is_initialized():
            iᵤ = self.all_gather(iᵤ).flatten(end_dim=1)
            ŷᵤ = self.all_gather(ŷᵤ).flatten(end_dim=1)
        iᵤ = iᵤ.cpu()
        ŷᵤ = ŷᵤ.cpu()
        self.criterionᵤ.Ŷ[iᵤ[ŷᵤ != -1]] = ŷᵤ[ŷᵤ != -1]

        self.train_acc.update(zₗ.softmax(dim=1), yₗ)
        return {'loss': loss,
                'detail': {'loss_l': lossₗ.detach(),
                           'loss_u': lossᵤ.detach(),
                           'mask': self.criterionᵤ.𝜇ₘₐₛₖ}}

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        self.ema.update_parameters(self.model)

    def training_epoch_end(self, outputs):
        loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.log('train/loss', loss, sync_dist=True)
        acc = self.train_acc.compute()
        self.log('train/acc', acc, rank_zero_only=True)
        self.train_acc.reset()

        loss = torch.stack([x['detail']['mask'] for x in outputs]).mean()
        self.log('detail/mask', loss, sync_dist=True)
        loss = torch.stack([x['detail']['loss_l'] for x in outputs]).mean()
        self.log('detail/loss_l', loss, sync_dist=True)
        loss = torch.stack([x['detail']['loss_u'] for x in outputs]).mean()
        self.log('detail/loss_u', loss, sync_dist=True)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        z = self.model(x)
        loss = self.criterionₗ(z, y)
        self.val_acc.update(z.softmax(dim=1), y)
        results = {'loss': loss}

        if self.hparams.model.get('ema'):
            z = self.ema(x)
            loss = self.criterionₗ(z, y)
            self.val_acc_ema.update(z.softmax(dim=1), y)
            results['loss/ema'] = loss

        return results

    def validation_epoch_end(self, outputs):
        loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.log('val/loss', loss, sync_dist=True)
        acc = self.val_acc.compute()
        self.log('val/acc', acc, rank_zero_only=True)
        self.val_acc.reset()

        if self.hparams.model.get('ema'):
            loss = torch.stack([x['loss/ema'] for x in outputs]).mean()
            self.log('val/loss/ema', loss, sync_dist=True)
            acc = self.val_acc_ema.compute()
            self.log('val/acc/ema', acc, rank_zero_only=True)
            self.val_acc_ema.reset()

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def test_epoch_end(self, outputs):
        return self.validation_epoch_end(outputs)

    def configure_optimizers(self):
        params = exclude_wd(self.model)
        optim = get_optim(params, **self.hparams.optimizer)
        sched = get_sched(optim, **self.hparams.scheduler)
        return {'optimizer': optim, 'lr_scheduler': {'scheduler': sched}}
