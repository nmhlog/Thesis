#!/bin/bash

# cp buff_agg/5050hierarchical_aggregation.cu lib/hais_ops/src/hierarchical_aggregation/hierarchical_aggregation.cu
# cp buff_agg/5050hierarchical_aggregation.cpp lib/hais_ops/src/hierarchical_aggregation/hierarchical_aggregation.cpp
# echo "Rebuild HAIS"
# bash rebuild_hais.sh>log_hais_rebuild.log
echo "Training Semantic"
echo "Training semantic_unet_50x50_ASPP"
CUDA_VISIBLE_DEVICES=4,5,6,7 bash dist_train-only_semantic.sh configs/training_only_semantic_segmentation/50x50_50/semantic_unet_50x50_ASPP.yaml 4 > semantic_unet_50x50_ASPP.log
echo "Training semantic_unet_50x50_ASPPv2"
CUDA_VISIBLE_DEVICES=4,5,6,7 bash dist_train-only_semantic.sh configs/training_only_semantic_segmentation/50x50_50/semantic_unet_50x50_ASPPv2.yaml 4 > semantic_unet_50x50_ASPPv2.log
echo "Training semantic_unet_50x50_ATTN_ASPP"
CUDA_VISIBLE_DEVICES=4,5,6,7 bash dist_train-only_semantic.sh configs/training_only_semantic_segmentation/50x50_50/semantic_unet_50x50_ATTN_ASPP.yaml 4 > semantic_unet_50x50_ATTN_ASPP.log

echo "Training Backbone"
echo "Training unet_ASPP"
CUDA_VISIBLE_DEVICES=4,5,6,7 bash dist_train_HAIS.sh configs/training_backbone/50x50_50/unet_ASPP.yaml 4 > unet_ASPP.log
echo "Training unet_ASPPv2"
CUDA_VISIBLE_DEVICES=4,5,6,7 bash dist_train_HAIS.sh configs/training_backbone/50x50_50/unet_ASPPv2.yaml 4 > unet_ASPPv2.log
echo "Training unet_ATTN_ASPP"
CUDA_VISIBLE_DEVICES=4,5,6,7 bash dist_train_HAIS.sh configs/training_backbone/50x50_50/unet_ATTN_ASPP.yaml 4 > unet_ATTN_ASPP.log

echo "Training Instance"
echo "Training HAIS_stpls3d_unet_aspp" 
CUDA_VISIBLE_DEVICES=4,5,6,7 bash dist_train_HAIS.sh configs/training_all/50X50_50/HAIS_stpls3d_unet_aspp.yaml 4 > HAIS_stpls3d_unet_aspp.log
echo "Training HAIS_stpls3d_unet_ASPPv2" 
CUDA_VISIBLE_DEVICES=4,5,6,7 bash dist_train_HAIS.sh configs/training_all/50X50_50/HAIS_stpls3d_unet_ASPPv2.yaml 4 > HAIS_stpls3d_unet_ASPPv2.log
echo "Training HAIS_stpls3d_unet_ATTN_ASPP" 
CUDA_VISIBLE_DEVICES=4,5,6,7 bash dist_train_HAIS.sh configs/training_all/50X50_50/HAIS_stpls3d_unet_ATTN_ASPP.yaml 4 > HAIS_stpls3d_unet_ATTN_ASPP.log

