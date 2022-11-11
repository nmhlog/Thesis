#!/bin/bash
# echo "Rebuild HAIS"
# cp buff_agg/CrossValidation/cv25050hierarchical_aggregation.cu lib/hais_ops/src/hierarchical_aggregation/hierarchical_aggregation.cu
# cp buff_agg/CrossValidation/cv25050hierarchical_aggregation.cpp lib/hais_ops/src/hierarchical_aggregation/hierarchical_aggregation.cpp

# echo "Rebuild HAIS"
# bash rebuild_hais.sh > rebuild_hais_cv2_5050.log
echo "Training semantic"
# CUDA_VISIBLE_DEVICES=2 python train_semantic_segmentation_model.py configs/cross_validation/5050/cv2_5050_semantic_unet.yaml
CUDA_VISIBLE_DEVICES=2 python train_semantic_segmentation_model.py configs/cross_validation/5050/cv2_5050_semantic_unet_ASPPv2.yaml --resume work_dirs/cv2_5050_semantic_unet_ASPPv2/epoch_37.pth

CUDA_VISIBLE_DEVICES=2 python train_semantic_segmentation_model.py configs/cross_validation/5050/cv2_5050_semantic_unet_ATTN_ASPP.yaml

CUDA_VISIBLE_DEVICES=2 python train_semantic_segmentation_model.py configs/cross_validation/5025/cv2_5025_semantic_unet.yaml