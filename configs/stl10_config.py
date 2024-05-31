from dataclasses import dataclass
from configs.base_config import BaseTrainingConfig
from configs.utils import register_config
from os.path import join

@dataclass
@register_config(name="stl10")
class STL10TrainingConfig(BaseTrainingConfig):

    output_dir = join("experiments", "cddpm-stl10-64")  # the model name locally and on the HF Hub
    
    # data information
    image_size = 64
    dataset_name = "stl10"

    # model architecture
    block_out_channels = (128, 256, 256, 256, 256)
    down_block_types = (
        "DownBlock2D",  # a regular ResNet downsampling block
        "DownBlock2D",
        "DownBlock2D",
        "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
        "DownBlock2D",
    )

    up_block_types = (
        "UpBlock2D",
        "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
    )
    