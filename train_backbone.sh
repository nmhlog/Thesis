#!/bin/bash

CUDA_VISIBLE_DEVICES=5 python train.py configs/training_backbone/50x50_50/unet_only_coordinated.yaml
CUDA_VISIBLE_DEVICES=5 python train.py configs/training_backbone/50x50_50/unet_ASPP_only_coordinated.yaml
CUDA_VISIBLE_DEVICES=5 python train.py configs/training_backbone/50x50_50/unet_ASPP_DICELOSS_only_coordinated.yaml
