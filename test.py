import os
import sys

from torchvision.datasets import CIFAR10, CIFAR100
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.utilities.argparse import parse_env_variables

from methods import NoisyFlexMatchClassifier
from torchvision.transforms import Compose
from weaver import get_transforms


def test(ckpt):
    trainer = Trainer(**vars(parse_env_variables(Trainer)))

    model = NoisyFlexMatchClassifier.load_from_checkpoint(ckpt)
    transform = Compose(get_transforms(model.hparams.transform['val']))

    dataset_name = model.hparams.dataset['name']
    batch_size = model.hparams.dataset['batch_size']

    Dataset = {
        'CIFAR10': CIFAR10,
        'CIFAR100': CIFAR100,
    }[dataset_name]

    dataset = Dataset(
        root=os.path.join('data', dataset_name),
        train=False,
        transform=transform
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size['clean'] + batch_size['noisy'],
        shuffle=False,
        num_workers=os.cpu_count(),
        pin_memory=True
    )

    trainer.test(model, dataloader)


if __name__ == "__main__":
    test(sys.argv[1])
