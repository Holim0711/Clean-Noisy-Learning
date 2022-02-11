import math
import torch
import pytorch_lightning as pl
from torchmetrics import Accuracy
from weaver.models import get_model
from weaver.optimizers import get_optim, exclude_wd
from weaver.schedulers import get_sched

__all__ = ['NoisyFlexMatchClassifier']


class AveragedModelWithBuffers(torch.optim.swa_utils.AveragedModel):
    def update_parameters(self, model):
        super().update_parameters(model)
        for a, b in zip(self.module.buffers(), model.buffers()):
            a.copy_(b.to(a.device))


class NoisyFlexMatchCrossEntropy(torch.nn.Module):
    def __init__(self, ỹ, T, temperature, threshold, epsilon=0., reduction='mean'):
        super().__init__()
        self.register_buffer('ỹ', torch.tensor(ỹ))
        self.register_buffer('T', torch.tensor(T))
        self.threshold = threshold
        self.temperature = temperature
        self.epsilon = epsilon
        self.reduction = reduction

        self.num_samples = len(ỹ)
        self.num_classes = len(T)
        self.register_buffer('ỹ_dist', self.ỹ.bincount() / self.num_samples)
        self.register_buffer('ŷ', torch.tensor([self.num_classes] * self.num_samples))
        self.𝜇ₘₐₛₖ = None

    def all_gather(self, x, world_size):
        x_list = [torch.zeros_like(x) for _ in range(world_size)]
        torch.distributed.all_gather(x_list, x)
        return torch.hstack(x_list)

    def forward(self, logits_s, logits_w, ỹ, i):
        ỹŷ = torch.zeros((self.num_classes, self.num_classes + 1))
        ỹŷ.index_put_((self.ỹ, self.ŷ), torch.ones(self.num_samples), True)
        ỹŷ = ỹŷ[:, :-1] + ỹŷ[:, -1:] * self.ỹ_dist.to(ỹŷ.device)
        ỹŷ /= ỹŷ.sum(axis=0)

        probs = torch.softmax(logits_w / self.temperature, dim=-1)
        probs = probs * self.T[:, ỹ].t() / (ỹŷ[ỹ].to(probs.device) + self.epsilon)
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
        self.ŷ[i[ŷ != -1]] = ŷ[ŷ != -1]

        loss = torch.nn.functional.cross_entropy(
            logits_s, targets, reduction='none') * masks
        self.𝜇ₘₐₛₖ = masks.float().mean().detach()

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
        self.log('train/mask', self.criterionᵤ.𝜇ₘₐₛₖ)
        self.train_acc.update(zₗ.softmax(dim=1), yₗ)
        return loss

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        self.ema.update_parameters(self.model)

    def training_epoch_end(self, outputs):
        acc = self.train_acc.compute()
        self.log('train/acc', acc)
        self.train_acc.reset()

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
