import torch
import torch.nn.functional as F
from torchmetrics.classification import MulticlassAccuracy
from .base import BaseModule

__all__ = ['NoisyFlexMatchClassifier']


class NoisyFlexMatchCrossEntropy(torch.nn.Module):

    def __init__(self, num_classes, num_samples, temperature, threshold):
        super().__init__()
        self.num_classes = num_classes
        self.num_samples = num_samples
        self.temperature = temperature
        self.threshold = threshold
        self.χ2 = float('inf')
        self.𝜇 = 0.0
        self.register_buffer('Ŷ', torch.tensor([num_classes] * num_samples))

    def register_transition_matrix(self, T):
        T = T if isinstance(T, torch.Tensor) else torch.tensor(T)
        assert T.shape == (self.num_classes, self.num_classes)
        assert torch.isclose(T.sum(axis=1), torch.tensor(1.)).all()
        self.register_buffer('T', T)

    def register_clean_label_info(self, Y):
        Y = Y if isinstance(Y, torch.Tensor) else torch.tensor(Y)
        assert (Y.unique() == torch.arange(self.num_classes)).all()
        self.register_buffer('Py', Y.bincount() / len(Y))

    def register_noisy_label_info(self, Ỹ):
        Ỹ = Ỹ if isinstance(Ỹ, torch.Tensor) else torch.tensor(Ỹ)
        assert len(Ỹ) == self.num_samples
        assert (Ỹ.unique() == torch.arange(self.num_classes)).all()
        self.register_buffer('Ỹ', Ỹ)
        self.register_buffer('Pỹ', Ỹ.bincount() / len(Ỹ))

    def prepare_constants(self):
        self.register_buffer('err_bnd', self.Pỹ * (1 - self.threshold) / self.Py.unsqueeze(-1))

    def forward(self, logits_s, logits_w, ỹ):
        T = torch.zeros((self.num_classes + 1, self.num_classes), dtype=int, device=self.Ŷ.device)
        T.index_put_((self.Ŷ, self.Ỹ), torch.tensor(1), accumulate=True)
        T = T[:-1] + T[-1] * self.Py.unsqueeze(-1) + 1
        T /= T.sum(axis=1, keepdims=True)

        α = self.T / T
        self.χ2 = ((α * self.T).sum(dim=-1) * self.Py).sum() - 1
        α = (α - 1) * ((self.T - T).abs() > self.err_bnd) + 1

        z = (logits_w / self.temperature).softmax(dim=-1) * α.t()[ỹ]
        c, ŷ = F.normalize(z, p=1).max(dim=-1)
        self.ŷ = torch.where(c > self.threshold, ŷ, -1)

        torch.use_deterministic_algorithms(False)
        β = self.Ŷ.bincount(minlength=self.num_classes + 1)
        torch.use_deterministic_algorithms(True)
        β = β / (2 * β.max() - β)

        mask = (c > self.threshold * β[ŷ])
        self.𝜇 = mask.float().mean()

        loss = F.cross_entropy(logits_s, ŷ, reduction='none')
        return (loss * mask).mean()


class NoisyFlexMatchModule(BaseModule):

    def __init__(self, **kwargs):
        super().__init__()
        self.criterionᶜ = torch.nn.CrossEntropyLoss()
        self.criterionⁿ = NoisyFlexMatchCrossEntropy(
            self.hparams['dataset']['num_classes'],
            self.hparams['dataset']['num_samples'],
            self.hparams.method['temperature'],
            self.hparams.method['threshold'])
        self.train_accuracy = MulticlassAccuracy(
            self.hparams['dataset']['num_classes'])

    def custom_init(self, T, Y, Ỹ):
        self.criterionⁿ.register_transition_matrix(T)
        self.criterionⁿ.register_clean_label_info(Y)
        self.criterionⁿ.register_noisy_label_info(Ỹ)
        self.criterionⁿ.prepare_constants()

    def training_step(self, batch, batch_idx):
        iᶜ, (xᶜ, yᶜ) = batch['clean']
        iⁿ, ((xʷ, xˢ), yⁿ) = batch['noisy']
        bᶜ, bⁿ = len(iᶜ), len(iⁿ)

        zᶜ, zʷ, zˢ = self.model(torch.cat((xᶜ, xʷ, xˢ))).split([bᶜ, bⁿ, bⁿ])

        lossᶜ = self.criterionᶜ(zᶜ, yᶜ)
        lossⁿ = self.criterionⁿ(zˢ, zʷ.detach(), yⁿ)
        loss = lossᶜ + lossⁿ

        self.log('train/loss', loss, on_step=False, on_epoch=True, sync_dist=True, batch_size=bᶜ + bⁿ)
        self.log('train/loss_c', lossᶜ, on_step=False, on_epoch=True, sync_dist=True, batch_size=bᶜ)
        self.log('train/loss_n', lossⁿ, on_step=False, on_epoch=True, sync_dist=True, batch_size=bⁿ)
        self.log('train/mask', self.criterionⁿ.𝜇, on_step=False, on_epoch=True, sync_dist=True, batch_size=bⁿ)
        self.log('train/chidiv', self.criterionⁿ.χ2, on_step=False, on_epoch=True, sync_dist=True, batch_size=bⁿ)
        self.train_accuracy.update(zᶜ, yᶜ)
        return {'loss': loss}

    def on_train_batch_end(self, outputs, batch, batch_idx):
        ĩ = batch['noisy'][0]
        ŷ = self.criterionⁿ.ŷ
        if torch.distributed.is_initialized():
            ĩ = self.all_gather(ĩ).flatten(end_dim=1)
            ŷ = self.all_gather(ŷ).flatten(end_dim=1)
        self.criterionⁿ.Ŷ[ĩ[ŷ != -1]] = ŷ[ŷ != -1]


    def training_epoch_end(self, outputs):
        self.log('train/acc', self.train_accuracy)
