#!/bin/bash
# echo "Rebuild HAIS 100x100"
# cp buff_agg/100100hierarchical_aggregation.cu lib/hais_ops/src/hierarchical_aggregation/hierarchical_aggregation.cu
# cp buff_agg/100100hierarchical_aggregation.cpp lib/hais_ops/src/hierarchical_aggregation/hierarchical_aggregation.cpp
# bash rebuild_hais.sh>log_hais_rebuild.log
# echo "Training backbone"
# echo "Training unet_100x100"
# CUDA_VISIBLE_DEVICES=1,2,3,4 bash dist_train_HAIS.sh configs/training_backbone/100x100/unet_100x100.yaml 4 
# echo "Training unet_100x100_ASPP"
# CUDA_VISIBLE_DEVICES=1,2,3,4 bash dist_train_HAIS.sh configs/training_backbone/100x100/unet_100x100_ASPP.yaml 4 
# echo "Training unet_100x100_ASPPv2"
# CUDA_VISIBLE_DEVICES=1,2,3,4 bash dist_train_HAIS.sh configs/training_backbone/100x100/unet_100x100_ASPPv2.yaml 4 
# echo "Training unet_100x100_ATTN_ASPP"
# CUDA_VISIBLE_DEVICES=1,2,3,4 bash dist_train_HAIS.sh configs/training_backbone/100x100/unet_100x100_ATTN_ASPP.yaml 4 

# echo "Rebuild HAIS 100x50"
# cp buff_agg/10050hierarchical_aggregation.cu lib/hais_ops/src/hierarchical_aggregation/hierarchical_aggregation.cu
# cp buff_agg/10050hierarchical_aggregation.cpp lib/hais_ops/src/hierarchical_aggregation/hierarchical_aggregation.cpp
# bash rebuild_hais.sh>log_hais_rebuild.log
# echo "Training Semantic"
# echo "Training semantic_unet"
# CUDA_VISIBLE_DEVICES=1,2,3,4 bash dist_train-only_semantic.sh configs/training_only_semantic_segmentation/100x100_50/semantic_unet_100x50.yaml 4 
# echo "Training semantic_unet_100x50_ASPP"
# CUDA_VISIBLE_DEVICES=1,2,3,4 bash dist_train-only_semantic.sh configs/training_only_semantic_segmentation/100x100_50/semantic_unet_100x50_ASPP.yaml 4 
# echo "Training semantic_unet_100x50_ASPPv2"
# CUDA_VISIBLE_DEVICES=1,2,3,4 bash dist_train-only_semantic.sh configs/training_only_semantic_segmentation/100x100_50/semantic_unet_100x50_ASPPv2.yaml 4 
# echo "Training semantic_unet_100x50_ATTN_ASPP"
# CUDA_VISIBLE_DEVICES=1,2,3,4 bash dist_train-only_semantic.sh configs/training_only_semantic_segmentation/100x100_50/semantic_unet_100x50_ATTN_ASPP.yaml 4 

# echo "Training backbone"
# echo "Training unet_100x50"
# CUDA_VISIBLE_DEVICES=1,2,3,4 bash dist_train_HAIS.sh configs/training_backbone/100x100_50/unet_100x50.yaml 4 
# echo "Training unet_100x50"
# CUDA_VISIBLE_DEVICES=1,2,3,4 bash dist_train_HAIS.sh configs/training_backbone/100x100_50/unet_100x50_ASPP.yaml 4 
# echo "Training backbone 100x50"
# echo "Training unet_100x50_ASPPv2"
# CUDA_VISIBLE_DEVICES=1,2,3,4 bash dist_train_HAIS.sh configs/training_backbone/100x100_50/unet_100x50_ASPPv2.yaml 4 --resume work_dirs/unet_100x50_ASPPv2/epoch_17.pth
# echo "Training unet_100x50_ATTN_ASPP"
# CUDA_VISIBLE_DEVICES=1,2,3,4 bash dist_train_HAIS.sh configs/training_backbone/100x100_50/unet_100x50_ATTN_ASPP.yaml 4 
# echo "Training Semantic 100x50"
# echo "Training semantic_unet 100x50"
# CUDA_VISIBLE_DEVICES=1,2,3,4 bash dist_train-only_semantic.sh configs/training_only_semantic_segmentation/100x100_50/semantic_unet_100x50.yaml 4 

# echo "Rebuild HAIS 100x100"
# cp buff_agg/100100hierarchical_aggregation.cu lib/hais_ops/src/hierarchical_aggregation/hierarchical_aggregation.cu
# cp buff_agg/100100hierarchical_aggregation.cpp lib/hais_ops/src/hierarchical_aggregation/hierarchical_aggregation.cpp
# bash rebuild_hais.sh>log_hais_rebuild.log
echo "Training Instance"
# echo "Training HAIS_stpls3d_unet_100x100"
# CUDA_VISIBLE_DEVICES=3,4,5,6 bash dist_train_HAIS.sh configs/training_all/100x100_100/HAIS_stpls3d_unet_100x100.yaml 4 --resume work_dirs/HAIS_stpls3d_unet_100x100/epoch_80.pth
# echo "Training HAIS_stpls3d_unet_aspp_100x100"
# CUDA_VISIBLE_DEVICES=3,4,5,6 bash dist_train_HAIS.sh configs/training_all/100x100_100/HAIS_stpls3d_unet_aspp_100x100.yaml 4 --resume work_dirs/HAIS_stpls3d_unet_aspp_100x100/epoch_71.pth
# echo "Training HAIS_stpls3d_unet_ASPPv2_100x100"
# CUDA_VISIBLE_DEVICES=3,4,5,6 bash dist_train_HAIS.sh configs/training_all/100x100_100/HAIS_stpls3d_unet_ASPPv2_100x100.yaml 4 --resume work_dirs/HAIS_stpls3d_unet_ASPPv2_100x100/epoch_103.pth
# echo "Training HAIS_stpls3d_unet_ATTN_ASPP_100x100"
# CUDA_VISIBLE_DEVICES=0,1,2,3 bash dist_train_HAIS.sh configs/training_all/100x100_100/HAIS_stpls3d_unet_ATTN_ASPP_100x100.yaml 4 --resume work_dirs/HAIS_stpls3d_unet_ATTN_ASPP_100x100/epoch_105.pth

# echo "Rebuild HAIS 100x50"
# cp buff_agg/10050hierarchical_aggregation.cu lib/hais_ops/src/hierarchical_aggregation/hierarchical_aggregation.cu
# cp buff_agg/10050hierarchical_aggregation.cpp lib/hais_ops/src/hierarchical_aggregation/hierarchical_aggregation.cpp
# bash rebuild_hais.sh>log_hais_rebuild.log

echo "Training Instance 100x50"
echo "Training HAIS_stpls3d_unet_100x50"
CUDA_VISIBLE_DEVICES=0,1,2,3 bash dist_train_HAIS.sh configs/training_all/100x100_50/HAIS_stpls3d_unet_100x50.yaml 4 --resume work_dirs/HAIS_stpls3d_unet_100x50/epoch_15.pth
echo "Training HAIS_stpls3d_unet_aspp_100x50"
CUDA_VISIBLE_DEVICES=0,1,2,3 bash dist_train_HAIS.sh configs/training_all/100x100_50/HAIS_stpls3d_unet_aspp_100x50.yaml 4 
echo "Training HAIS_stpls3d_unet_ASPPv2_100x50"
CUDA_VISIBLE_DEVICES=0,1,2,3 bash dist_train_HAIS.sh configs/training_all/100x100_50/HAIS_stpls3d_unet_ASPPv2_100x50.yaml 4 
echo "Training HAIS_stpls3d_unet_ATTN_ASPP_100x50"
CUDA_VISIBLE_DEVICES=0,1,2,3 bash dist_train_HAIS.sh configs/training_all/100x100_50/HAIS_stpls3d_unet_ATTN_ASPP_100x50.yaml 4 
