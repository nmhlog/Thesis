#!/bin/bash

bash dist_train-only_semantic.sh configs/training_all/50x50_25/HAIS_stpls3d_unet_50X25.yaml 4
bash dist_train-only_semantic.sh configs/training_all/50x50_25/HAIS_stpls3d_unet_aspp_50X25.yaml 4

bash dist_train_HAIS.sh configs/training_backbone/50x50_25/unet_50x25.yaml 4
bash dist_train_HAIS.sh configs/training_backbone/50x50_25/unet_50x25.yaml 4

bash dist_train_HAIS.sh configs/training_all/50x50_25/HAIS_stpls3d_unet_50X25.yaml 4
bash dist_train_HAIS.sh configs/training_all/50x50_25/HAIS_stpls3d_unet_aspp_50X25.yaml 4