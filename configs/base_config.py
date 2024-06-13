from dataclasses import dataclass
from os.path import join


@dataclass
class BaseTrainingConfig:

    # metadata info
    output_dir = join("experiments", "base")

    # data information
    dataset_name = None
    image_size = 32
    image_chan = 3
    class_condition_dim = 10

    # model information
    layers_per_block = 2
    block_out_channels = (128, 256, 256, 256)
    down_block_types = (
        "DownBlock2D",  # a regular ResNet downsampling block
        "DownBlock2D",
        "DownBlock2D",
        "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
    )

    up_block_types = (
        "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
    )


    # training and evaluation
    cond_drop_rate = 0.3 # condtion drop rate for classifier-free guidance
    train_batch_size = 128
    eval_batch_size = 10
    num_epochs = 200
    gradient_accumulation_steps = 1
    learning_rate = 1e-4
    lr_warmup_steps = 500
    save_image_epochs = 20
    save_model_epochs = 50
    mixed_precision = "fp16"  # `no` for float32, `fp16` for automatic mixed precision
    # output_dir = "cddpm-base-32"  # the model name locally and on the HF Hub
    num_inference_steps = 500

    overwrite_output_dir = False  # overwrite the old model when re-running the notebook
    seed = 0


