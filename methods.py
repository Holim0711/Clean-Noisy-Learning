import math
import torch
import pytorch_lightning as pl
from torchmetrics import Accuracy
from weaver.models import get_model
from weaver.optimizers import get_optim, exclude_wd
from weaver.schedulers import get_sched
from utils import plot_confusion_matrix

__all__ = ['NoisyFlexMatchClassifier']


class AveragedModelWithBuffers(torch.optim.swa_utils.AveragedModel):
    def update_parameters(self, model):
        super().update_parameters(model)
        for a, b in zip(self.module.buffers(), model.buffers()):
            a.copy_(b.to(a.device))


class NoisyFlexMatchCrossEntropy(torch.nn.Module):
    def __init__(self, ỹ, T, threshold, temperature, reduction='mean'):
        super().__init__()

        if not isinstance(ỹ, torch.Tensor):
            ỹ = torch.tensor(ỹ)
        if not isinstance(T, torch.Tensor):
            T = torch.tensor(T)

        self.C = len(ỹ.unique())                    # |C|
        self.D = len(ỹ)                             # |D|
        self.ŷ = torch.tensor([self.C] * self.D)    # All ŷₖ = ∅ at first
        self.ỹ = ỹ
        self.T = T                                  # P(ỹ|y)
        self.Pỹ = ỹ.bincount() / len(ỹ)             # P(ỹ)

        self.threshold = threshold
        self.temperature = temperature
        self.reduction = reduction

        # Monitoring Variables
        self.𝜇 = None
        self.M = None

    def all_gather(self, x, world_size):
        x_list = [torch.zeros_like(x) for _ in range(world_size)]
        torch.distributed.all_gather(x_list, x)
        return torch.hstack(x_list)

    def forward(self, logits_s, logits_w, ỹ, i):
        M = torch.zeros((self.C + 1, self.C), dtype=torch.long)
        M.index_put_((self.ŷ, self.ỹ), torch.tensor(1), True)
        M = M[:-1] + (M[-1] + self.Pỹ) * self.Pỹ
        M = M / M.sum(axis=1)
        α = (self.T / M).to(logits_w.device)

        probs = torch.softmax(logits_w / self.temperature, dim=-1)
        probs *= α[:, ỹ].t()
        probs /= probs.sum(dim=-1, keepdim=True)
        max_probs, targets = probs.max(dim=-1)

        β = self.ŷ.cpu().bincount()
        β = β / β.max()
        β = β / (2 - β)
        β = β.to(max_probs.device)
        masks = max_probs > self.threshold * β[targets]

        ŷ = torch.where(max_probs > self.threshold, targets, -1)
        if torch.distributed.is_initialized():
            world_size = torch.distributed.get_world_size()
            ŷ = self.all_gather(ŷ, world_size)
            i = self.all_gather(i, world_size)
        i = i[ŷ != -1].cpu()
        ŷ = ŷ[ŷ != -1].cpu()
        self.ŷ[i] = ŷ

        loss = torch.nn.functional.cross_entropy(
            logits_s, targets, reduction='none') * masks

        # Monitoring Variables
        self.𝜇 = masks.float().mean().item()
        self.M = M

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


def change_bn(model, momentum):
    if isinstance(model, torch.nn.BatchNorm2d):
        model.momentum = 1 - momentum
    else:
        for children in model.children():
            change_bn(children, momentum)


class NoisyFlexMatchClassifier(pl.LightningModule):

    def __init__(self, ỹ, T, **kwargs):
        super().__init__()
        for k in kwargs:
            self.save_hyperparameters(k)
        self.steps_per_epoch = math.ceil(len(ỹ) / self.hparams.dataset['batch_sizes']['noisy'])

        self.model = get_model(**self.hparams.model['backbone'])
        self.criterionₗ = torch.nn.CrossEntropyLoss()
        self.criterionᵤ = NoisyFlexMatchCrossEntropy(ỹ, T, **self.hparams.model['loss_u'])
        self.train_acc = Accuracy()
        self.valid_acc = Accuracy()

        change_bn(self.model, self.hparams.model['momentum'])

        def avg_fn(averaged_model_parameter, model_parameter, num_averaged):
            m = self.hparams.model['momentum']
            return m * averaged_model_parameter + (1 - m) * model_parameter
        self.ema = AveragedModelWithBuffers(self.model, avg_fn=avg_fn)

    def training_step(self, batch, batch_idx):
        xₗ, yₗ = batch['clean']
        iᵤ, ((ˢxᵤ, ʷxᵤ), ỹ) = batch['noisy']

        z = self.model(torch.cat((xₗ, ˢxᵤ, ʷxᵤ)))
        zₗ = z[:xₗ.shape[0]]
        ˢzᵤ, ʷzᵤ = z[xₗ.shape[0]:].chunk(2)
        del z

        lossₗ = self.criterionₗ(zₗ, yₗ)
        lossᵤ = self.criterionᵤ(ˢzᵤ, ʷzᵤ.detach(), ỹ, iᵤ)
        loss = lossₗ + lossᵤ

        self.log('train/loss', loss)
        self.log('train/loss_l', lossₗ)
        self.log('train/loss_u', lossᵤ)
        self.log('train/mask', self.criterionᵤ.𝜇)
        self.train_acc.update(zₗ.softmax(dim=1), yₗ)
        return loss

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        self.ema.update_parameters(self.model)

    def training_epoch_end(self, outputs):
        acc = self.train_acc.compute()
        self.log('train/acc', acc)
        self.train_acc.reset()

        logger = self.logger.experiment

        # logging T
        if self.current_epoch == 0:
            if len(self.criterionᵤ.T) <= 20:
                fig = plot_confusion_matrix(self.criterionᵤ.T)
                logger.add_figure('T', fig, 0)
            else:
                logger.add_image('T', self.criterionᵤ.T, 0, dataformats='HW')

        # logging M
        if len(self.criterionᵤ.T) <= 20:
            fig = plot_confusion_matrix(self.criterionᵤ.M)
            logger.add_figure('M', fig, self.current_epoch)
        else:
            logger.add_image('M', self.criterionᵤ.M, self.current_epoch,
                             dataformats='HW')

    def validation_step(self, batch, batch_idx):
        x, y = batch
        z = self.ema(x)
        loss = self.criterionₗ(z, y)
        self.log('val/loss', loss, on_step=False, on_epoch=True, sync_dist=True)
        self.valid_acc.update(z.softmax(dim=1), y)
        return loss

    def validation_epoch_end(self, outputs):
        acc = self.valid_acc.compute()
        self.log('val/acc', acc)
        self.valid_acc.reset()

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def test_epoch_end(self, outputs):
        return self.validation_epoch_end(outputs)

    def configure_optimizers(self):
        params = exclude_wd(self.model)
        optim = get_optim(params, **self.hparams.optimizer)
        sched = get_sched(optim, **self.hparams.scheduler)
        sched.extend(self.steps_per_epoch)
        return {'optimizer': optim,
                'lr_scheduler': {'scheduler': sched, 'interval': 'step'}}
