#!/bin/bash

bash dist_train.sh configs/training_different_feature/unet.yaml 4
bash dist_train.sh configs/training_different_feature/unet_ASPP.yaml 4
bash dist_train.sh configs/training_different_feature/unet_ASPP_DICELOSS.yaml 4

bash dist_train.sh configs/training_different_feature/unet_coordinated.yaml 4
bash dist_train.sh configs/training_different_feature/unet_ASPP_coordinated.yaml 4
bash dist_train.sh configs/training_different_feature/unet_ASPP_DICELOSS_coordinated.yaml 4

bash dist_train_HAIS.sh configs/training_all/50X50_50/HAIS_stpls3d_unet.yaml 4
bash dist_train_HAIS.sh configs/training_all/50X50_50/HAIS_stpls3d_unet_aspp.yaml 4
bash dist_train_HAIS.sh configs/training_all/50X50_50/HAIS_stpls3d_unet_aspp_diceloss.yaml 4

bash dist_train.sh configs/training_all/50X50_50/softgroup_stpls3d_unet.yaml 4
bash dist_train.sh configs/training_all/50X50_50/softgroup_stpls3d_unet_aspp.yaml 4
bash dist_train.sh configs/training_all/50X50_50/softgroup_stpls3d_unet_aspp_diceloss.yaml 4
