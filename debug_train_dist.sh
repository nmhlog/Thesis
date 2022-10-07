#!/bin/bash

CUDA_VISIBLE_DEVICES=4,5,6,7 bash dist_train_HAIS.sh configs/training_backbone/100x100_50/unet_100x50.yaml 4 
CUDA_VISIBLE_DEVICES=4,5,6,7 bash dist_train_HAIS.sh configs/training_backbone/100x100_50/unet_100x50_ASPP.yaml 4

CUDA_VISIBLE_DEVICES=4,5,6,7 bash dist_train_HAIS.sh configs/training_all/100x100_50/HAIS_stpls3d_unet_100x50.yaml 4 
CUDA_VISIBLE_DEVICES=4,5,6,7 bash dist_train_HAIS.sh configs/training_all/100x100_50/HAIS_stpls3d_unet_aspp_100x50.yaml 4 
