#!/bin/bash
# echo "Rebuild HAIS"
# cp buff_agg/100100hierarchical_aggregation.cu lib/hais_ops/src/hierarchical_aggregation/hierarchical_aggregation.cu
# cp buff_agg/100100hierarchical_aggregation.cpp lib/hais_ops/src/hierarchical_aggregation/hierarchical_aggregation.cpp
# bash rebuild_hais.sh

echo "Training Semantic"
echo "Training semantic_unet_100x100"
CUDA_VISIBLE_DEVICES=2 python train_semantic_segmentation_model.py configs/training_only_semantic_segmentation/100x100/semantic_unet_100x100.yaml  

# echo "Rebuild HAIS"
# cp buff_agg/10050hierarchical_aggregation.cu lib/hais_ops/src/hierarchical_aggregation/hierarchical_aggregation.cu
# cp buff_agg/10050hierarchical_aggregation.cpp lib/hais_ops/src/hierarchical_aggregation/hierarchical_aggregation.cpp
# bash rebuild_hais.sh
# echo "Training Semantic"
# echo "Training semantic_unet"
# CUDA_VISIBLE_DEVICES=2 python train_semantic_segmentation_model.py configs/training_only_semantic_segmentation/100x100_50/semantic_unet_100x50.yaml 
