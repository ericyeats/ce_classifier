from dataclasses import dataclass
from configs.base_config import BaseTrainingConfig
from configs.utils import register_config
from os.path import join

@dataclass
@register_config(name="cifar10")
class CIFAR10TrainingConfig(BaseTrainingConfig):

    dataset_name = "cifar10"
    output_dir = join("experiments", "cddpm-cifar10-32")  # the model name locally and on the HF Hub
    
    

