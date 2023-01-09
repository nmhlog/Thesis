# CUDA_VISIBLE_DEVICES=4 python test-HAIS.py configs/training_only_semantic_segmentation/50x50_50/semantic_unet_50x50.yaml pretrain/5050/semantic_unet_50x50/epoch_20.pth --out out/5050/semantic_unet_50x50/ 
# CUDA_VISIBLE_DEVICES=4 python test-HAIS.py configs/training_only_semantic_segmentation/50x50_50/semantic_unet_50x50_ASPP.yaml pretrain/5050/semantic_unet_50x50_ASPP/epoch_20.pth --out out/5050/semantic_unet_50x50_ASPP/
# CUDA_VISIBLE_DEVICES=4 python test-HAIS.py configs/training_only_semantic_segmentation/50x50_50/semantic_unet_50x50_ASPPv2.yaml pretrain/5050/semantic_unet_50x50_ASPPv2/epoch_20.pth --out out/5050/semantic_unet_50x50_ASPPv2/
# CUDA_VISIBLE_DEVICES=4 python test-HAIS.py configs/training_only_semantic_segmentation/50x50_50/semantic_unet_50x50_ATTN_ASPP.yaml pretrain/5050/semantic_unet_50x50_ATTN_ASPP/epoch_20.pth --out out/5050/semantic_unet_50x50_ATTN_ASPP/


# CUDA_VISIBLE_DEVICES=4 python test-HAIS.py configs/training_only_semantic_segmentation/50x50_25/semantic_unet_50x25.yaml pretrain/5025/semantic_unet_50x25/epoch_20.pth --out out/5025/semantic_unet_50x25/ 
# CUDA_VISIBLE_DEVICES=4 python test-HAIS.py configs/training_only_semantic_segmentation/50x50_25/semantic_unet_50x25_ASPP.yaml pretrain/5025/semantic_unet_50x25_ASPP/epoch_20.pth --out out/5025/semantic_unet_50x25_ASPP/
# CUDA_VISIBLE_DEVICES=4 python test-HAIS.py configs/training_only_semantic_segmentation/50x50_25/semantic_unet_50x25_ASPPv2.yaml pretrain/5025/semantic_unet_50x25_ASPPv2/epoch_20.pth --out out/5025/semantic_unet_50x25_ASPPv2/
# CUDA_VISIBLE_DEVICES=4 python test-HAIS.py configs/training_only_semantic_segmentation/50x50_25/semantic_unet_50x25_ATTN_ASPP.yaml pretrain/5025/semantic_unet_50x25_ATTN_ASPP/epoch_20.pth --out out/5025/semantic_unet_50x25_ATTN_ASPP/

# CUDA_VISIBLE_DEVICES=4 python test-HAIS.py configs/training_only_semantic_segmentation/100x100/semantic_unet_100x50.yaml pretrain/100100/semantic__unet_100x100/epoch_20.pth --out out/100100/semantic_unet_100x100/

# CUDA_VISIBLE_DEVICES=4 python test-HAIS.py configs/training_only_semantic_segmentation/100x100_50/semantic_unet_100x50.yaml pretrain/10050/semantic_unet_100x50/epoch_20.pth --out out/10050/semantic_unet_100x50/
echo "Rebuild HAIS"
cp buff_agg/5050hierarchical_aggregation.cu lib/hais_ops/src/hierarchical_aggregation/hierarchical_aggregation.cu
cp buff_agg/5050hierarchical_aggregation.cpp lib/hais_ops/src/hierarchical_aggregation/hierarchical_aggregation.cpp
echo "rebuild_hais_5050"
bash rebuild_hais.sh> rebuild_hais_5050.log
echo "instance segmentation 5050"
CUDA_VISIBLE_DEVICES=4 python test-HAIS.py configs/training_all/50X50_50/HAIS_stpls3d_unet.yaml pretrain/5050/HAIS_stpls3d_unet/epoch_84.pth --out out/inst/5050/HAIS_stpls3d_unet/ 
# CUDA_VISIBLE_DEVICES=4 python test-HAIS.py configs/training_all/50X50_50/HAIS_stpls3d_unet_aspp.yaml pretrain/5050/HAIS_stpls3d_unet_aspp/epoch_108.pth --out out/5050/HAIS_stpls3d_unet_aspp/
# CUDA_VISIBLE_DEVICES=4 python test-HAIS.py configs/training_all/50X50_50/HAIS_stpls3d_unet_ASPPv2.yaml pretrain/5050/HAIS_stpls3d_unet_ASPPv2/epoch_108.pth --out out/5050/HAIS_stpls3d_unet_ASPPv2/
CUDA_VISIBLE_DEVICES=4 python test-HAIS.py configs/training_all/50X50_50/HAIS_stpls3d_unet_ATTN_ASPP.yaml pretrain/5050/HAIS_stpls3d_unet_ATTN_ASPP/epoch_100.pth --out out/inst/5050/HAIS_stpls3d_unet_ATTN_ASPP/
zip -r HAIS_stpls3d_unet_ATTN_ASPP.zip out/inst/5050/HAIS_stpls3d_unet_ATTN_ASPP/*
echo "Rebuild HAIS"

cp buff_agg/5025hierarchical_aggregation.cu lib/hais_ops/src/hierarchical_aggregation/hierarchical_aggregation.cu
cp buff_agg/5025hierarchical_aggregation.cpp lib/hais_ops/src/hierarchical_aggregation/hierarchical_aggregation.cpp
echo "Rebuild HAIS"
bash rebuild_hais.sh> rebuild_hais_5025.log&

# echo "instance segmentation 5025"
CUDA_VISIBLE_DEVICES=4 python test-HAIS.py configs/training_all/50x50_25/HAIS_stpls3d_unet_50X25.yaml pretrain/5025/HAIS_stpls3d_unet_50X25/epoch_96.pth --out out/inst/5025/HAIS_stpls3d_unet_50X25/ 
# CUDA_VISIBLE_DEVICES=4 python test-HAIS.py configs/training_all/50x50_25/HAIS_stpls3d_unet_aspp_50X25.yaml pretrain/5025/HAIS_stpls3d_unet_aspp_50X25/epoch_108.pth --out out/5025/HAIS_stpls3d_unet_aspp_50X25/
# CUDA_VISIBLE_DEVICES=4 python test-HAIS.py configs/training_all/50x50_25/HAIS_stpls3d_unet_ASPPv2_50x25.yaml pretrain/5025/HAIS_stpls3d_unet_ASPPv2_50x25/epoch_108.pth --out out/5025/HAIS_stpls3d_unet_ASPPv2_50x25/
CUDA_VISIBLE_DEVICES=4 python test-HAIS.py configs/training_all/50x50_25/HAIS_stpls3d_unet_ATTN_ASPP_50x25.yaml pretrain/5025/HAIS_stpls3d_unet_ATTN_ASPP_50x25/epoch_104.pth --out out/inst/5025/HAIS_stpls3d_unet_ATTN_ASPP_50x25/
zip -r HAIS_stpls3d_unet_ATTN_ASPP_50x25.zip out/inst/5025/HAIS_stpls3d_unet_ATTN_ASPP_50x25/*
# CUDA_VISIBLE_DEVICES=4 python test-HAIS.py configs/training_all/100x100_100/HAIS_stpls3d_unet_100x100.yaml pretrain/100100/HAIS_stpls3d_unet_100x100/epoch_108.pth --out out/100100/HAIS_stpls3d_unet_100x100/

# CUDA_VISIBLE_DEVICES=4 python test-HAIS.py configs/training_all/100x100_50/HAIS_stpls3d_unet_100x50.yaml pretrain/10050/HAIS_stpls3d_unet_100x50/epoch_108.pth --out out/10050/HAIS_stpls3d_unet_100x50/
# # backbone

# CUDA_VISIBLE_DEVICES=4 python test-HAIS.py configs/training_backbone/50x50_50/unet.yaml pretrain/5050/unet/epoch_20.pth --out out/5050/unet/ 
# CUDA_VISIBLE_DEVICES=4 python test-HAIS.py configs/training_backbone/50x50_50/unet_ASPP.yaml pretrain/5050/unet_ASPP/epoch_20.pth --out out/5050/unet_ASPP/
# CUDA_VISIBLE_DEVICES=4 python test-HAIS.py configs/training_backbone/50x50_50/unet_ASPPv2.yaml pretrain/5050/unet_ASPPv2/epoch_20.pth --out out/5050/unet_ASPPv2/
# CUDA_VISIBLE_DEVICES=4 python test-HAIS.py configs/training_backbone/50x50_50/unet_ATTN_ASPP.yaml pretrain/5050/unet_ATTN_ASPP/epoch_20.pth --out out/5050/unet_ATTN_ASPP/


# CUDA_VISIBLE_DEVICES=4 python test-HAIS.py configs/training_backbone/50x50_25/unet_50x25.yaml pretrain/5025/unet_50x25/epoch_20.pth --out out/5025/unet_50x25/ 
# CUDA_VISIBLE_DEVICES=4 python test-HAIS.py configs/training_backbone/50x50_25/unet_50x25_ASPP.yaml pretrain/5025/unet_50x25_ASPP/epoch_20.pth --out out/5025/unet_50x25_ASPP/
# CUDA_VISIBLE_DEVICES=4 python test-HAIS.py configs/training_backbone/50x50_25/unet_50x25_ASPPv2.yaml pretrain/5025/unet_50x25_ASPPv2/epoch_20.pth --out out/5025/unet_50x25_ASPPv2/
# CUDA_VISIBLE_DEVICES=4 python test-HAIS.py configs/training_backbone/50x50_25/unet_50x25_ATTN_ASPP.yaml pretrain/5025/unet_50x25_ATTN_ASPP/epoch_20.pth --out out/5025/unet_50x25_ATTN_ASPP/

# CUDA_VISIBLE_DEVICES=4 python test-HAIS.py configs/training_backbone/100x100/unet_100x100.yaml pretrain/100100/unet_100x100/epoch_20.pth --out out/100100/unet_100x100/

# CUDA_VISIBLE_DEVICES=4 python test-HAIS.py configs/training_backbone/100x100_50/unet_100x50.yaml pretrain/10050/unet_100x50/epoch_20.pth --out out/10050/unet_100x50/
