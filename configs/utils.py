from configs.base_config import BaseTrainingConfig
from torchvision.datasets import CIFAR10, STL10, SVHN

from torchvision import transforms as T
from typing import Tuple, Union
from diffusers.models.unets.unet_2d import UNet2DModel
from torch.utils import data as D


_CONFIGS = {}

def register_config(cls=None, *, name: Union[str, None] = None):

    def _register(cls):

        local_name = name if name is not None else cls.__name__
        if local_name in _CONFIGS:
            raise ValueError(f"Already registered config with name {local_name}")
        
        _CONFIGS[local_name] = cls

        return cls

    if cls is None:
        return _register
    else:
        return _register(cls)

def get_config(name: str) -> BaseTrainingConfig:

    if name in _CONFIGS.keys():
        return _CONFIGS[name]
    else:
        raise ValueError(f"Config name {name} not registered")


# quick dataset lookup implementation

def get_dataset(cfg: BaseTrainingConfig) -> Tuple[D.Dataset, D.Dataset]:

    # get the train & test sets from cfg
    name = cfg.dataset_name
    train_dataset = test_dataset = None
    if name == "cifar10":
        train_tform = T.Compose([T.RandomHorizontalFlip(), T.ToTensor(), T.Normalize(0.5, 0.5)])
        test_tform = T.Compose([T.ToTensor(), T.Normalize(0.5, 0.5)])
        train_dataset = CIFAR10(root='~/data/cifarpy', train=True, transform=train_tform)
        test_dataset = CIFAR10(root='~/data/cifarpy', train=False, transform=test_tform)
    elif name == "svhn":
        train_tform = T.Compose([T.ToTensor(), T.Normalize(0.5, 0.5)])
        test_tform = T.Compose([T.ToTensor(), T.Normalize(0.5, 0.5)])
        train_dataset = SVHN(root='~/data/svhn', split='train', transform=train_tform)
        test_dataset = SVHN(root='~/data/svhn', split='test', transform=test_tform)
    elif name == "stl10":
        train_tform = T.Compose([T.RandomHorizontalFlip(), T.Resize(cfg.image_size), \
                                 T.ToTensor(), T.Normalize(0.5, 0.5)])
        test_tform = T.Compose([T.Resize(cfg.image_size), T.ToTensor(), T.Normalize(0.5, 0.5)])
        train_dataset = STL10(root='~/data/stl10', split='train', transform=train_tform)
        test_dataset = STL10(root='~/data/stl10', split='test', transform=test_tform)
        print("Allocating additional data to STL Train...")
        # create a new train/test dataset split. first, split the test dataset. second, concat some test to training
        test_to_train_size = 7000
        test_to_train_data = D.Subset(test_dataset, range(test_to_train_size))
        train_dataset = D.ConcatDataset([train_dataset, test_to_train_data])
        test_dataset = D.Subset(test_dataset, range(test_to_train_size, len(test_dataset)))
    else:
        raise NotImplementedError(f"Dataset {name} not implemented")

    return train_dataset, test_dataset

def get_model(cfg: BaseTrainingConfig) -> UNet2DModel:

    model = UNet2DModel(
        sample_size=cfg.image_size,  # the target image resolution
        in_channels=cfg.image_chan,  # the number of input channels, 3 for RGB images
        out_channels=cfg.image_chan,  # the number of output channels
        layers_per_block=cfg.layers_per_block,  # how many ResNet layers to use per UNet block
        block_out_channels=cfg.block_out_channels,  # the number of output channels for each UNet block
        down_block_types=cfg.down_block_types,
        up_block_types=cfg.up_block_types,
        num_class_embeds=cfg.class_condition_dim + 1, # classifier-free guidance
    )
    
    return model