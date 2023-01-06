import os
from math import ceil
from typing import Callable, Union
from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader
from torchvision.datasets import CIFAR10, CIFAR100
from weaver.datasets import RandomSubset, IndexedDataset
from noisy_cifar import NoisyCIFAR10, NoisyCIFAR100


class BaseDataModule(LightningDataModule):

    def __init__(
        self,
        root: str,
        transforms: dict[str, Callable],
        batch_sizes: dict[str, int],
    ):
        super().__init__()
        self.root = root
        self.splits = ['clean', 'noisy', 'val']
        self.transforms = {k: transforms[k] for k in self.splits}
        self.batch_sizes = {k: batch_sizes[k] for k in self.splits}

    def get_raw_dataset(self, split: str):
        raise NotImplementedError(f"self.get_raw_dataset('{split}')")

    def setup(self, stage=None):
        datasets = [self.get_raw_dataset(k) for k in self.splits]
        datasets[0] = IndexedDataset(datasets[0])
        datasets[1] = IndexedDataset(datasets[1])

        n0 = len(datasets[0]) / self.batch_sizes[self.splits[0]]
        n1 = len(datasets[1]) / self.batch_sizes[self.splits[1]]
        m = ceil(n1 / (n0 * 2))
        datasets[0] = ConcatDataset([datasets[0]] * m)

        self.datasets = dict(zip(self.splits, datasets))

    def get_dataloader(
        self,
        split: str,
        shuffle: bool = True,
        num_workers: int = os.cpu_count(),
        pin_memory: bool = True
    ):
        return DataLoader(
            self.datasets[split],
            self.batch_sizes[split],
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    def train_dataloader(self):
        return {self.splits[0]: self.get_dataloader(self.splits[0]),
                self.splits[1]: self.get_dataloader(self.splits[1])}

    def val_dataloader(self):
        return self.get_dataloader(self.splits[2], shuffle=False)


class CIFAR10DataModule(BaseDataModule):

    def __init__(
        self,
        root: str,
        num_clean: int,
        noise_type: str,
        noise_level: Union[float, str],
        transforms: dict[str, Callable],
        batch_sizes: dict[str, int],
        random_seed: int = 0,
    ):
        super().__init__(root, transforms, batch_sizes)
        self.num_clean = num_clean
        self.noise_type = noise_type
        self.noise_level = noise_level
        self.random_seed = random_seed

    def prepare_data(self):
        CIFAR10(self.root, download=True)

    def get_raw_dataset(self, split: str):
        transform = self.transforms.get(split)
        if split == 'clean':
            return RandomSubset(CIFAR10(self.root, transform=transform),
                                self.num_clean,
                                class_balanced=True,
                                random_seed=self.random_seed)
        elif split == 'noisy':
            return NoisyCIFAR10(self.root, self.noise_type, self.noise_level,
                                transform=transform,
                                random_seed=self.random_seed)
        elif split == 'val':
            return CIFAR10(self.root, train=False, transform=transform)
        else:
            raise KeyError(split)


class CIFAR100DataModule(BaseDataModule):

    def __init__(
        self,
        root: str,
        num_clean: int,
        noise_type: str,
        noise_level: Union[float, str],
        transforms: dict[str, Callable],
        batch_sizes: dict[str, int],
        random_seed: int = 0,
    ):
        super().__init__(root, transforms, batch_sizes)
        self.num_clean = num_clean
        self.noise_type = noise_type
        self.noise_level = noise_level
        self.random_seed = random_seed

    def prepare_data(self):
        CIFAR100(self.root, download=True)

    def get_raw_dataset(self, split: str):
        transform = self.transforms.get(split)
        if split == 'clean':
            return RandomSubset(CIFAR100(self.root, transform=transform),
                                self.num_clean,
                                class_balanced=True,
                                random_seed=self.random_seed)
        elif split == 'noisy':
            return NoisyCIFAR100(self.root, self.noise_type, self.noise_level,
                                 transform=transform,
                                 random_seed=self.random_seed)
        elif split == 'val':
            return CIFAR100(self.root, train=False, transform=transform)
        else:
            raise KeyError(split)
