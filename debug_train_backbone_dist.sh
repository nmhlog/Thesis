#!/bin/bash
cp buff_agg/5050hierarchical_aggregation.cu lib/hais_ops/src/hierarchical_aggregation/hierarchical_aggregation.cu
cp buff_agg/5050hierarchical_aggregation.cpp lib/hais_ops/src/hierarchical_aggregation/hierarchical_aggregation.cpp
bash rebuild_hais.sh > log_hais_installation.log
bash dist_train-only_semantic.sh configs/training_only_semantic_segmentation/50x50_50/semantic_unet_50x50_ASPP.yaml 4

