#!/bin/bash

CUDA_VISIBLE_DEVICES=5 python train.py configs/training_different_feature/unet.yaml
CUDA_VISIBLE_DEVICES=5 python train.py configs/training_different_feature/unet_ASPP.yaml
CUDA_VISIBLE_DEVICES=5 python train.py configs/training_different_feature/unet_ASPP_DICELOSS.yaml


