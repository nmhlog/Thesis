#!/bin/bash
echo "Rebuild HAIS 100x100"
cp buff_agg/100100hierarchical_aggregation.cu lib/hais_ops/src/hierarchical_aggregation/hierarchical_aggregation.cu
cp buff_agg/100100hierarchical_aggregation.cpp lib/hais_ops/src/hierarchical_aggregation/hierarchical_aggregation.cpp
bash rebuild_hais.sh>log_hais_rebuild.log
echo "Training backbone"
echo "Training unet_100x100"
CUDA_VISIBLE_DEVICES=1,3,4,5 bash dist_train_HAIS.sh configs/training_backbone/100x100/unet_100x100.yaml 4 
echo "Training unet_100x100_ASPP"
CUDA_VISIBLE_DEVICES=1,3,4,5 bash dist_train_HAIS.sh configs/training_backbone/100x100/unet_100x100_ASPP.yaml 4 
echo "Training unet_100x100_ASPPv2"
CUDA_VISIBLE_DEVICES=1,3,4,5 bash dist_train_HAIS.sh configs/training_backbone/100x100/unet_100x100_ASPPv2.yaml 4 
echo "Training unet_100x100_ATTN_ASPP"
CUDA_VISIBLE_DEVICES=1,3,4,5 bash dist_train_HAIS.sh configs/training_backbone/100x100/unet_100x100_ATTN_ASPP.yaml 4 

echo "Rebuild HAIS 100x50"
cp buff_agg/10050hierarchical_aggregation.cu lib/hais_ops/src/hierarchical_aggregation/hierarchical_aggregation.cu
cp buff_agg/10050hierarchical_aggregation.cpp lib/hais_ops/src/hierarchical_aggregation/hierarchical_aggregation.cpp
bash rebuild_hais.sh>log_hais_rebuild.log
echo "Training Semantic"
echo "Training semantic_unet"
CUDA_VISIBLE_DEVICES=1,3,4,5 bash dist_train-only_semantic.sh configs/training_only_semantic_segmentation/100x100_50/semantic_unet_100x50.yaml4 
echo "Training semantic_unet_100x50_ASPP"
CUDA_VISIBLE_DEVICES=1,3,4,5 bash dist_train-only_semantic.sh configs/training_only_semantic_segmentation/100x100_50/semantic_unet_100x50_ASPP.yaml 4 
echo "Training semantic_unet_100x50_ASPPv2"
CUDA_VISIBLE_DEVICES=1,3,4,5 bash dist_train-only_semantic.sh configs/training_only_semantic_segmentation/100x100_50/semantic_unet_100x50_ASPPv2.yaml 4 
echo "Training semantic_unet_100x50_ATTN_ASPP"
CUDA_VISIBLE_DEVICES=1,3,4,5 bash dist_train-only_semantic.sh configs/training_only_semantic_segmentation/100x100_50/semantic_unet_100x50_ATTN_ASPP.yaml 4 

echo "Training backbone"
echo "Training unet_100x50"
CUDA_VISIBLE_DEVICES=1,3,4,5 bash dist_train_HAIS.sh configs/training_backbone/100x100_50/unet_100x50.yaml 4 
echo "Training unet_100x50"
CUDA_VISIBLE_DEVICES=1,3,4,5 bash dist_train_HAIS.sh configs/training_backbone/100x100_50/unet_100x50_ASPP.yaml 4 
echo "Training unet_100x50_ASPPv2"
CUDA_VISIBLE_DEVICES=1,3,4,5 bash dist_train_HAIS.sh configs/training_backbone/100x100_50/unet_100x50_ASPPv2.yaml 4 
echo "Training unet_100x50_ATTN_ASPP"
CUDA_VISIBLE_DEVICES=1,3,4,5 bash dist_train_HAIS.sh configs/training_backbone/100x100_50/unet_100x50_ATTN_ASPP.yaml 4 
 
