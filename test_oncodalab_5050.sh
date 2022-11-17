# cp buff_agg/5050hierarchical_aggregation.cu lib/hais_ops/src/hierarchical_aggregation/hierarchical_aggregation.cu
# cp buff_agg/5050hierarchical_aggregation.cpp lib/hais_ops/src/hierarchical_aggregation/hierarchical_aggregation.cpp
# echo "Rebuild HAIS 50-50"
# bash rebuild_hais.sh>log_hais_rebuild5050.log

CUDA_VISIBLE_DEVICES=2 python test-HAIS-oncodalab.py configs/Test/50x50_50/HAIS_stpls3d_unet.yaml pretrain/5050/HAIS_stpls3d_unet/epoch_84.pth --out out/HAIS_stpls3d_unet/
CUDA_VISIBLE_DEVICES=2 python test-HAIS-oncodalab.py configs/Test/50x50_50/HAIS_stpls3d_unet_ATTN_ASPP.yaml pretrain/5050/HAIS_stpls3d_unet_ATTN_ASPP/epoch_100.pth --out out/HAIS_stpls3d_unet_ATTN_ASPP/