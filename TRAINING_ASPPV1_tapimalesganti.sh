#!/bin/bash

cp buff_agg/5050hierarchical_aggregation.cu lib/hais_ops/src/hierarchical_aggregation/hierarchical_aggregation.cu
cp buff_agg/5050hierarchical_aggregation.cpp lib/hais_ops/src/hierarchical_aggregation/hierarchical_aggregation.cpp
echo "Rebuild HAIS"
# bash rebuild_hais.sh>log_hais_rebuild.log
# echo "Training semantic_unet"
# echo "Training semantic_unet ATTN_ASPP"
# CUDA_VISIBLE_DEVICES=3,4,5,6 bash dist_train-only_semantic.sh configs/training_only_semantic_segmentation/50x50_50/semantic_unet_50x50_ATTN_ASPP.yaml 4 
echo "Training Backbone"
echo "Training unet_ATTN_ASPP"
CUDA_VISIBLE_DEVICES=3,4,5,6 bash dist_train_HAIS.sh configs/training_backbone/50x50_50/unet_ATTN_ASPP.yaml 4 
echo "Training Instance"
echo "Training HAIS_stpls3d_unet_ATTN_ASPP" 
CUDA_VISIBLE_DEVICES=3,4,5,6 bash dist_train_HAIS.sh configs/training_all/50X50_50/HAIS_stpls3d_unet_ATTN_ASPP.yaml 4

cp buff_agg/5025hierarchical_aggregation.cu lib/hais_ops/src/hierarchical_aggregation/hierarchical_aggregation.cu
cp buff_agg/5025hierarchical_aggregation.cpp lib/hais_ops/src/hierarchical_aggregation/hierarchical_aggregation.cpp
echo "Rebuild HAIS"
bash rebuild_hais.sh>log_hais_rebuild.log
echo "Training semantic_unet ATTN_ASPP"
CUDA_VISIBLE_DEVICES=3,4,5,6 bash dist_train-only_semantic.sh configs/training_only_semantic_segmentation/50x50_25/semantic_unet_50x25_ATTN_ASPP.yaml 4 
echo "Training Backbone"
echo "Training unet_ATTN_ASPP"
CUDA_VISIBLE_DEVICES=3,4,5,6 bash dist_train_HAIS.sh configs/training_backbone/50x50_25/unet_50x25_ATTN_ASPP.yaml 4 
echo "Training Instance"
echo "Training HAIS_stpls3d_unet_ATTN_ASPP" 
CUDA_VISIBLE_DEVICES=3,4,5,6 bash dist_train_HAIS.sh configs/training_all/50x50_25/HAIS_stpls3d_unet_ATTN_ASPP_50x25.yaml 4


