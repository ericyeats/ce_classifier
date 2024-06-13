#!/bin/bash

accelerate launch train_diffusion_model.py --config cifar10 --ckpt_start ./experiments/cddpm-cifar10-32/ 