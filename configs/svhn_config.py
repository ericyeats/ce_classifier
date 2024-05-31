from dataclasses import dataclass
from configs.base_config import BaseTrainingConfig
from configs.utils import register_config
from os.path import join

@dataclass
@register_config(name="svhn")
class SVHNTrainingConfig(BaseTrainingConfig):

    dataset_name = "svhn"
    output_dir = join("experiments", "cddpm-svhn-32")  # the model name locally and on the HF Hub
    

