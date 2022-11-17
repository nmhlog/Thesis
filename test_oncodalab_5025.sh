cp buff_agg/5025hierarchical_aggregation.cu lib/hais_ops/src/hierarchical_aggregation/hierarchical_aggregation.cu
cp buff_agg/5025hierarchical_aggregation.cpp lib/hais_ops/src/hierarchical_aggregation/hierarchical_aggregation.cpp
echo "Rebuild HAIS 50-25"
bash rebuild_hais.sh>log_hais_rebuild5025.log

CUDA_VISIBLE_DEVICES=2 python test-HAIS-oncodalab.py configs/Test/50x50_25/HAIS_stpls3d_unet_50x25.yaml pretrain/5025/HAIS_stpls3d_unet_50X25/epoch_96.pth --out out/HAIS_stpls3d_unet_50X25/
CUDA_VISIBLE_DEVICES=2 python test-HAIS-oncodalab.py configs/Test/50x50_25/HAIS_stpls3d_unet_ATTN_ASPP_50x25.yaml pretrain/5025/HAIS_stpls3d_unet_ATTN_ASPP_50x25/epoch_104.pth --out out/HAIS_stpls3d_unet_ATTN_ASPP_50x25/