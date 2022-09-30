#!/bin/bash

bash dist_train.sh configs/training_different_feature/unet.yaml 4
bash dist_train.sh configs/training_different_feature/unet_ASPP.yaml 4
bash dist_train.sh configs/training_different_feature/unet_ASPP_DICELOSS.yaml 4
