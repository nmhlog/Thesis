#!/bin/bash

cp buff_agg/5050hierarchical_aggregation.cu lib/hais_ops/src/hierarchical_aggregation/hierarchical_aggregation.cu
cp buff_agg/5050hierarchical_aggregation.cpp lib/hais_ops/src/hierarchical_aggregation/hierarchical_aggregation.cpp
bash rebuild_hais.sh>log_hais_rebuild.log

CUDA_VISIBLE_DEVICES=4,5,6,7 bash dist_train-only_semantic.sh configs/training_only_semantic_segmentation/50x50_50/semantic_unet_50x50_ASPP.yaml 4 
CUDA_VISIBLE_DEVICES=4,5,6,7 bash dist_train-only_semantic.sh configs/training_only_semantic_segmentation/50x50_50/semantic_unet_50x50_ASPPv2.yaml 4 
CUDA_VISIBLE_DEVICES=4,5,6,7 bash dist_train-only_semantic.sh configs/training_only_semantic_segmentation/50x50_50/semantic_unet_50x50_ATTN_ASPP.yaml 4 

CUDA_VISIBLE_DEVICES=4,5,6,7 bash dist_train_HAIS.sh configs/training_backbone/50x50_50/unet_ASPP.yaml 4 
CUDA_VISIBLE_DEVICES=4,5,6,7 bash dist_train_HAIS.sh configs/training_backbone/50x50_50/unet_ASPPv2.yaml 4 
CUDA_VISIBLE_DEVICES=4,5,6,7 bash dist_train_HAIS.sh configs/training_backbone/50x50_50/unet _ATTN_ASPP.yaml 4 
 
CUDA_VISIBLE_DEVICES=4,5,6,7 bash dist_train_HAIS.sh configs/training_all/50X50_50/HAIS_stpls3d_unet_aspp.yaml 4 
CUDA_VISIBLE_DEVICES=4,5,6,7 bash dist_train_HAIS.sh configs/training_all/50X50_50/HAIS_stpls3d_unet_ASPPv2.yaml 4 
CUDA_VISIBLE_DEVICES=4,5,6,7 bash dist_train_HAIS.sh configs/training_all/50X50_50/HAIS_stpls3d_unet_ATTN_ASPP.yaml 4 
