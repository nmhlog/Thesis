#!/bin/bash

CUDA_VISIBLE_DEVICES=4,5,6,7 bash dist_train_HAIS.sh configs/training_all/50x50_25/HAIS_stpls3d_unet_aspp_50X25.yaml 4 --resume work_dirs/HAIS_stpls3d_unet_aspp_50X25/epoch_64.pth
