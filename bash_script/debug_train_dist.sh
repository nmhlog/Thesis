#!/bin/bash

CUDA_VISIBLE_DEVICES=4,5,6,7 bash dist_train_HAIS.sh configs/training_all/100x100_50/HAIS_stpls3d_unet_100x50.yaml 4 --resume work_dirs/HAIS_stpls3d_unet_100x50/epoch_7.pth
 
