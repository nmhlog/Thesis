#!/bin/bash

bash dist_train.sh configs/training_different_feature/unet_only_coordinated.yaml 4
bash dist_train.sh configs/training_different_feature/unet_ASPP_only_coordinated.yaml 4
bash dist_train.sh configs/training_different_feature/unet_ASPP_DICELOSS_only_coordinated.yaml 4
