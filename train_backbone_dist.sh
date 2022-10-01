#!/bin/bash

bash dist_train-only_semantic.sh configs/training_only_semantic_segmentation/50x50_50/semantic_unet_50x50.yaml 4
bash dist_train-only_semantic.sh configs/training_only_semantic_segmentation/50x50_50/semantic_unet_50x50.yaml 4
bash dist_train-only_semantic.sh configs/training_only_semantic_segmentation/50x50_50/semantic_unet_50x50.yaml 4

bash dist_train_HAIS.sh configs/training_backbone/50x50_50/unet.yaml 4
bash dist_train_HAIS.sh configs/training_backbone/50x50_50/unet_ASPP.yaml 4
bash dist_train_HAIS.sh configs/training_backbone/50x50_50/unet_ASPP_DICELOSS.yaml 4

bash dist_train.sh configs/training_all/50X50_50/HAIS_stpls3d_unet.yaml 4
bash dist_train.sh configs/training_all/50X50_50/HAIS_stpls3d_unet_aspp.yaml 4
bash dist_train.sh configs/training_all/50X50_50/HAIS_stpls3d_unet_aspp_diceloss.yaml 4


