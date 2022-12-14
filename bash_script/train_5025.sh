#!/bin/bash

# cp buff_agg/5025hierarchical_aggregation.cu lib/hais_ops/src/hierarchical_aggregation/hierarchical_aggregation.cu
# cp buff_agg/5025hierarchical_aggregation.cpp lib/hais_ops/src/hierarchical_aggregation/hierarchical_aggregation.cpp
# echo "Rebuild HAIS"
# bash rebuild_hais.sh
# echo "Rebuild HAIS"
# bash rebuild_hais.sh
# CUDA_VISIBLE_DEVICES=2 python train_semantic_segmentation_model.py configs/training_only_semantic_segmentation/50x50_25/semantic_unet_50x25.yaml 
# CUDA_VISIBLE_DEVICES=2 python train_semantic_segmentation_model.py configs/training_only_semantic_segmentation/50x50_25/semantic_unet_50x25_ASPPv2.yaml 
# CUDA_VISIBLE_DEVICES=2 python train_semantic_segmentation_model.py configs/training_only_semantic_segmentation/50x50_25/semantic_unet_50x25_ATTN_ASPP.yaml 
CUDA_VISIBLE_DEVICES=2 python train_semantic_segmentation_model.py configs/training_only_semantic_segmentation/50x50_25/semantic_unet_50x25_ASPP.yaml 
# CUDA_VISIBLE_DEVICES=2 python train_semantic_segmentation_model.py configs/training_only_semantic_segmentation/50x50_25/semantic_unet_50x25_ATTN_ASPPv1.yaml 
