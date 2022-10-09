#!/bin/bash

cp buff_agg/5025hierarchical_aggregation.cu lib/hais_ops/src/hierarchical_aggregation/hierarchical_aggregation.cu
cp buff_agg/5025hierarchical_aggregation.cpp lib/hais_ops/src/hierarchical_aggregation/hierarchical_aggregation.cpp
bash rebuild_hais.sh>log_hais_rebuild.log

CUDA_VISIBLE_DEVICES=4,5,6,7 bash dist_train-only_semantic.sh configs/training_only_semantic_segmentation/50x50_25/semantic_unet_50x25_ASPP.yaml 4 
CUDA_VISIBLE_DEVICES=4,5,6,7 bash dist_train-only_semantic.sh configs/training_only_semantic_segmentation/50x50_25/semantic_unet_50x25_ASPPv2.yaml 4 
CUDA_VISIBLE_DEVICES=4,5,6,7 bash dist_train-only_semantic.sh configs/training_only_semantic_segmentation/50x50_25/semantic_unet_50x25_ATTN_ASPP.yaml 4 

CUDA_VISIBLE_DEVICES=4,5,6,7 bash dist_train_HAIS.sh configs/training_backbone/50x50_25/unet_50x25_ASPP.yaml 4 
CUDA_VISIBLE_DEVICES=4,5,6,7 bash dist_train_HAIS.sh configs/training_backbone/50x50_25/unet_50x25_ASPPv2.yaml 4 
CUDA_VISIBLE_DEVICES=4,5,6,7 bash dist_train_HAIS.sh configs/training_backbone/50x50_25/unet_50x25_ATTN_ASPP.yaml 4 
 
CUDA_VISIBLE_DEVICES=4,5,6,7 bash dist_train_HAIS.sh configs/training_all/50x50_25/HAIS_stpls3d_unet_aspp_50x25.yaml 4 
CUDA_VISIBLE_DEVICES=4,5,6,7 bash dist_train_HAIS.sh configs/training_all/50x50_25/HAIS_stpls3d_unet_ASPPv2_50x25.yaml 4 
CUDA_VISIBLE_DEVICES=4,5,6,7 bash dist_train_HAIS.sh configs/training_all/50x50_25/HAIS_stpls3d_unet_ATTN_ASPP_50x25.yaml 4 

