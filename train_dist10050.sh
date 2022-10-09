#!/bin/bash

cp buff_agg/10050hierarchical_aggregation.cu lib/hais_ops/src/hierarchical_aggregation/hierarchical_aggregation.cu
cp buff_agg/10050hierarchical_aggregation.cpp lib/hais_ops/src/hierarchical_aggregation/hierarchical_aggregation.cpp
bash rebuild_hais.sh>log_hais_rebuild.log

CUDA_VISIBLE_DEVICES=4,5,6,7 bash dist_train-only_semantic.sh configs/training_only_semantic_segmentation/100x100_50/semantic__unet_100x50_ASPP.yaml 4 
CUDA_VISIBLE_DEVICES=4,5,6,7 bash dist_train-only_semantic.sh configs/training_only_semantic_segmentation/100x100_50/semantic_unet_100x50_ASPPv2.yaml 4 
CUDA_VISIBLE_DEVICES=4,5,6,7 bash dist_train-only_semantic.sh configs/training_only_semantic_segmentation/100x100_50/semantic_unet_100x50_ATTN_ASPP.yaml 4 

CUDA_VISIBLE_DEVICES=4,5,6,7 bash dist_train_HAIS.sh configs/training_backbone/100x100_50/unet_100x50_ASPP.yaml 4 
CUDA_VISIBLE_DEVICES=4,5,6,7 bash dist_train_HAIS.sh configs/training_backbone/100x100_50/unet_100x50_ASPPv2.yaml 4 
CUDA_VISIBLE_DEVICES=4,5,6,7 bash dist_train_HAIS.sh configs/training_backbone/100x100_50/unet_100x50_ATTN_ASPP.yaml 4 
 
CUDA_VISIBLE_DEVICES=4,5,6,7 bash dist_train_HAIS.sh configs/training_all/100x100_50/HAIS_stpls3d_unet_aspp_100x50.yaml 4 
CUDA_VISIBLE_DEVICES=4,5,6,7 bash dist_train_HAIS.sh configs/training_all/100x100_50/HAIS_stpls3d_unet_ASPPv2_100x50.yaml 4 
CUDA_VISIBLE_DEVICES=4,5,6,7 bash dist_train_HAIS.sh configs/training_all/100x100_50/HAIS_stpls3d_unet_ATTN_ASPP_100x50.yaml 4 
