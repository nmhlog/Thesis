#!/bin/bash

CUDA_VISIBLE_DEVICES=5 python train.py configs/training_different_feature/unet.yaml
CUDA_VISIBLE_DEVICES=5 python train.py configs/training_different_feature/unet_ASPP.yaml
CUDA_VISIBLE_DEVICES=5 python train.py configs/training_different_feature/unet_ASPP_DICELOSS.yaml

CUDA_VISIBLE_DEVICES=5 python train.py configs/training_different_feature/unet_coordinated.yaml
CUDA_VISIBLE_DEVICES=5 python train.py configs/training_different_feature/unet_ASPP_coordinated.yaml
CUDA_VISIBLE_DEVICES=5 python train.py configs/training_different_feature/unet_ASPP_DICELOSS_coordinated.yaml

