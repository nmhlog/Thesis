#!/bin/bash
echo "Rebuild HAIS"
cp buff_agg/5050hierarchical_aggregation.cu lib/hais_ops/src/hierarchical_aggregation/hierarchical_aggregation.cu
cp buff_agg/5050hierarchical_aggregation.cpp lib/hais_ops/src/hierarchical_aggregation/hierarchical_aggregation.cpp
echo "Rebuild HAIS"
# bash rebuild_hais.sh
# CUDA_VISIBLE_DEVICES=2 python train_semantic_segmentation_model.py configs/training_only_semantic_segmentation/50x50_50/semantic_unet_50x50.yaml --resume work_dirs/semantic_unet_50x50/epoch_20.pth
CUDA_VISIBLE_DEVICES=2 python train_semantic_segmentation_model.py configs/training_only_semantic_segmentation/50x50_50/semantic_unet_50x50_ASPPv2.yaml --resume work_dirs/semantic_unet_50x50_ASPPv2/epoch_20.pth
CUDA_VISIBLE_DEVICES=2 python train_semantic_segmentation_model.py configs/training_only_semantic_segmentation/50x50_50/semantic_unet_50x50_ATTN_ASPP.yaml 
# CUDA_VISIBLE_DEVICES=2 python train_semantic_segmentation_model.py configs/training_only_semantic_segmentation/50x50_50/semantic_unet_50x50_ASPP.yaml --resume work_dirs/semantic_unet_50x50_ASPP/epoch_20.pth


# CUDA_VISIBLE_DEVICES=2 python train-HAIS.py configs/training_all/50X50_50/HAIS_stpls3d_unet_ATTN_ASPPv1.yaml

# echo "Training Semantic"
# echo "Training semantic_unet"
# CUDA_VISIBLE_DEVICES=3,4,5,6 bash dist_train-only_semantic.sh configs/training_only_semantic_segmentation/100x100_50/semantic_unet_100x50.yaml4 
# echo "Training semantic_unet_100x50_ASPP"
# CUDA_VISIBLE_DEVICES=3,4,5,6 bash dist_train-only_semantic.sh configs/training_only_semantic_segmentation/100x100_50/semantic_unet_100x50_ASPP.yaml 4 
# echo "Training semantic_unet_100x50_ASPPv2"
# CUDA_VISIBLE_DEVICES=3,4,5,6 bash dist_train-only_semantic.sh configs/training_only_semantic_segmentation/100x100_50/semantic_unet_100x50_ASPPv2.yaml 4 
# echo "Training semantic_unet_100x50_ATTN_ASPP"
# CUDA_VISIBLE_DEVICES=3,4,5,6 bash dist_train-only_semantic.sh configs/training_only_semantic_segmentation/100x100_50/semantic_unet_100x50_ATTN_ASPP.yaml 4 

# echo "Training backbone"
# echo "Training unet_100x50"
# CUDA_VISIBLE_DEVICES=3,4,5,6 bash dist_train_HAIS.sh configs/training_backbone/100x100_50/unet_100x50.yaml 4 
# echo "Training unet_100x50"
# CUDA_VISIBLE_DEVICES=3,4,5,6 bash dist_train_HAIS.sh configs/training_backbone/100x100_50/unet_100x50_ASPP.yaml 4 
# echo "Training unet_100x50_ASPPv2"
# CUDA_VISIBLE_DEVICES=3,4,5,6 bash dist_train_HAIS.sh configs/training_backbone/100x100_50/unet_100x50_ASPPv2.yaml 4 
# echo "Training unet_100x50_ATTN_ASPP"
# CUDA_VISIBLE_DEVICES=3,4,5,6 bash dist_train_HAIS.sh configs/training_backbone/100x100_50/unet_100x50_ATTN_ASPP.yaml 4 
 
# echo "Training Instance"
# echo "Training HAIS_stpls3d_unet_100x50"
# CUDA_VISIBLE_DEVICES=3,4,5,6 bash dist_train_HAIS.sh configs/training_all/100x100_50/HAIS_stpls3d_unet_100x50.yaml 4 
# echo "Training HAIS_stpls3d_unet_aspp_100x50"
# CUDA_VISIBLE_DEVICES=3,4,5,6 bash dist_train_HAIS.sh configs/training_all/100x100_50/HAIS_stpls3d_unet_aspp_100x50.yaml 4 
# echo "Training HAIS_stpls3d_unet_ASPPv2_100x50"
# CUDA_VISIBLE_DEVICES=3,4,5,6 bash dist_train_HAIS.sh configs/training_all/100x100_50/HAIS_stpls3d_unet_ASPPv2_100x50.yaml 4 
# echo "Training HAIS_stpls3d_unet_ATTN_ASPP_100x50"
# CUDA_VISIBLE_DEVICES=3,4,5,6 bash dist_train_HAIS.sh configs/training_all/100x100_50/HAIS_stpls3d_unet_ATTN_ASPP_100x50.yaml 4 
