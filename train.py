import hydra
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.utilities.argparse import parse_env_variables
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning_lite import seed_everything
from torchvision.transforms import Compose, Lambda
from weaver import get_transforms

from methods import NoisyFlexMatchModule
from datasets import CIFAR10DataModule, CIFAR100DataModule


@hydra.main(config_path='configs', version_base=None)
def main(config):
    seed_everything(config['random_seed'])

    trainer = Trainer(
        **vars(parse_env_variables(Trainer)),
        callbacks=[
            ModelCheckpoint(monitor='val/acc', mode='max'),
            LearningRateMonitor(),
        ]
    )

    transform_w = Compose(get_transforms(config['transform']['weak']))
    transform_s = Compose(get_transforms(config['transform']['strong']))
    transform_v = Compose(get_transforms(config['transform']['val']))
    transforms = {
        'clean': transform_w,
        'noisy': Lambda(lambda x: (transform_w(x), transform_s(x))),
        'val': transform_v,
    }

    batch_sizes = config['batch_size']
    batch_sizes = {k: v // trainer.num_devices for k, v in batch_sizes.items()}

    if config['dataset']['name'] == 'CIFAR10':
        dm = CIFAR10DataModule(
            config['dataset']['root'],
            config['dataset']['num_clean'],
            config['dataset']['noise_type'],
            config['dataset']['noise_level'],
            transforms=transforms,
            batch_sizes=batch_sizes,
            random_seed=config['dataset']['random_seed'])
    elif config['dataset']['name'] == 'CIFAR100':
        dm = CIFAR100DataModule(
            config['dataset']['root'],
            config['dataset']['num_clean'],
            config['dataset']['noise_type'],
            config['dataset']['noise_level'],
            transforms=transforms,
            batch_sizes=batch_sizes,
            random_seed=config['dataset']['random_seed'])

    if config['method']['name'] == 'NoisyFlexMatch':
        model = NoisyFlexMatchModule(**config)
        path= 'transition_matrix/{name}/{noise_type}-{noise_level}-{random_seed}.pth'.format(**config['dataset'])
        T = torch.load(path)['inv_cur_h']
        Y = [int(y) for _, y in dm.get_raw_dataset('clean')]
        Ỹ = [int(y) for _, y in dm.get_raw_dataset('noisy')]
        model.custom_init(T, Y, Ỹ)

    trainer.fit(model, dm)


if __name__ == "__main__":
    main()
