#!/bin/bash

bash dist_train.sh configs/training_all/50X50_50/softgroup_stpls3d_unet_aspp_diceloss.yaml 4
bash dist_train.sh configs/training_all/50X50_50/softgroup_stpls3d_unet_aspp.yaml 4
bash dist_train.sh configs/training_all/50X50_50/softgroup_stpls3d_unet_aspp_diceloss.yaml 4

