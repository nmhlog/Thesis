#!/bin/bash

cp buff_agg/100100hierarchical_aggregation.cu lib/hais_ops/src/hierarchical_aggregation/hierarchical_aggregation.cu
cp buff_agg/100100hierarchical_aggregation.cpp lib/hais_ops/src/hierarchical_aggregation/hierarchical_aggregation.cpp
bash rebuild_hais.sh>log_hais_rebuild.log

CUDA_VISIBLE_DEVICES=4,5,6,7 bash dist_train-only_semantic.sh configs/training_only_semantic_segmentation/100x100/semantic__unet_100x100_ASPP.yaml 4 
CUDA_VISIBLE_DEVICES=4,5,6,7 bash dist_train-only_semantic.sh configs/training_only_semantic_segmentation/100x100/semantic__unet_100x100_ASPPv2.yaml 4 
CUDA_VISIBLE_DEVICES=4,5,6,7 bash dist_train-only_semantic.sh configs/training_only_semantic_segmentation/100x100/semantic__unet_100x100_ATTN_ASPP.yaml 4 

CUDA_VISIBLE_DEVICES=4,5,6,7 bash dist_train_HAIS.sh configs/training_backbone/100x100/unet_100x100_ASPP.yaml 4 
CUDA_VISIBLE_DEVICES=4,5,6,7 bash dist_train_HAIS.sh configs/training_backbone/100x100/unet_100x100_ASPPv2.yaml 4 
CUDA_VISIBLE_DEVICES=4,5,6,7 bash dist_train_HAIS.sh configs/training_backbone/100x100/unet_100x100_ATTN_ASPP.yaml 4 
 
CUDA_VISIBLE_DEVICES=4,5,6,7 bash dist_train_HAIS.sh configs/training_all/100x100_100/HAIS_stpls3d_unet_aspp_100x100.yaml 4 
CUDA_VISIBLE_DEVICES=4,5,6,7 bash dist_train_HAIS.sh configs/training_all/100x100_100/HAIS_stpls3d_unet_ASPPv2_100x100.yaml 4 
CUDA_VISIBLE_DEVICES=4,5,6,7 bash dist_train_HAIS.sh configs/training_all/100x100_100/HAIS_stpls3d_unet_ATTN_ASPP_100x100.yaml 4 
