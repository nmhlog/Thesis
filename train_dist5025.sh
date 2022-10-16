#!/bin/bash

# cp buff_agg/5025hierarchical_aggregation.cu lib/hais_ops/src/hierarchical_aggregation/hierarchical_aggregation.cu
# cp buff_agg/5025hierarchical_aggregation.cpp lib/hais_ops/src/hierarchical_aggregation/hierarchical_aggregation.cpp
# echo "Rebuild HAIS"
# bash rebuild_hais.sh>log_hais_rebuild.log
# echo "Training Semantic"
# echo "Training semantic_unet ASPP"
# CUDA_VISIBLE_DEVICES=3,4,5,6 bash dist_train-only_semantic.sh configs/training_only_semantic_segmentation/50x50_25/semantic_unet_50x25_ASPP.yaml 4 
# echo "Training semantic_unet ASPPv2"
# CUDA_VISIBLE_DEVICES=3,4,5,6 bash dist_train-only_semantic.sh configs/training_only_semantic_segmentation/50x50_25/semantic_unet_50x25_ASPPv2.yaml 4 
# echo "Training semantic_unet ATTN_ASPP"
# CUDA_VISIBLE_DEVICES=3,4,5,6 bash dist_train-only_semantic.sh configs/training_only_semantic_segmentation/50x50_25/semantic_unet_50x25_ATTN_ASPP.yaml 4 
# echo "Training Backbone"
# echo "Training unet_ASPP"
# CUDA_VISIBLE_DEVICES=3,4,5,6 bash dist_train_HAIS.sh configs/training_backbone/50x50_25/unet_50x25_ASPP.yaml 4 
# echo "Training unet_ASPPv2"
# CUDA_VISIBLE_DEVICES=3,4,5,6 bash dist_train_HAIS.sh configs/training_backbone/50x50_25/unet_50x25_ASPPv2.yaml 4 
# echo "Training unet_ATTN_ASPP"
# CUDA_VISIBLE_DEVICES=3,4,5,6 bash dist_train_HAIS.sh configs/training_backbone/50x50_25/unet_50x25_ATTN_ASPP.yaml 4 
# echo "Training Instance"
# echo "Training HAIS_stpls3d_unet_aspp" 
# CUDA_VISIBLE_DEVICES=3,4,5,6 bash dist_train_HAIS.sh configs/training_all/50x50_25/HAIS_stpls3d_unet_aspp_50X25.yaml 4 --resume work_dirs/HAIS_stpls3d_unet_aspp_50X25/epoch_80.pth
echo "Training HAIS_stpls3d_unet_ASPPv2" 
# CUDA_VISIBLE_DEVICES=3,4,5,6 bash dist_train_HAIS.sh configs/training_all/50x50_25/HAIS_stpls3d_unet_ASPPv2_50x25.yaml 4 --resume work_dirs/HAIS_stpls3d_unet_ASPPv2_50x25/epoch_88.pth
echo "Training HAIS_stpls3d_unet_ATTN_ASPP" 
CUDA_VISIBLE_DEVICES=3,4,5,6 bash dist_train_HAIS.sh configs/training_all/50x50_25/HAIS_stpls3d_unet_ATTN_ASPP_50x25.yaml 4 --resume work_dirs/HAIS_stpls3d_unet_ATTN_ASPP_50x25/epoch_88.pth


