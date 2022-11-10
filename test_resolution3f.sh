
echo "Semantic Segmentasi 5025"
CUDA_VISIBLE_DEVICES=2 python test-Semantic.py configs/training_only_semantic_segmentation/50x50_25/semantic_unet_50x25.yaml pretrain/5025/semantic_unet_50x25/epoch_20.pth --out out/5025/semantic_unet_50x25/ 
CUDA_VISIBLE_DEVICES=2 python test-Semantic.py configs/training_only_semantic_segmentation/50x50_25/semantic_unet_50x25_ASPPv2.yaml pretrain/5025/semantic_unet_50x25_ASPPv2/epoch_16.pth --out out/5025/semantic_unet_50x25_ASPPv2/
CUDA_VISIBLE_DEVICES=2 python test-Semantic.py configs/training_only_semantic_segmentation/50x50_25/semantic_unet_50x25_ATTN_ASPP.yaml pretrain/5025/semantic_unet_50x25_ATTN_ASPP/epoch_16.pth --out out/5025/semantic_unet_50x25_ATTN_ASPP/

echo "instance segmentation 5025"
CUDA_VISIBLE_DEVICES=2 python test-HAIS.py configs/training_all/50x50_25/HAIS_stpls3d_unet_50X25.yaml pretrain/5025/HAIS_stpls3d_unet_50X25/epoch_108.pth --out out/5025/HAIS_stpls3d_unet_50X25/ 
CUDA_VISIBLE_DEVICES=2 python test-HAIS.py configs/training_all/50x50_25/HAIS_stpls3d_unet_ASPPv2_50x25.yaml pretrain/5025/HAIS_stpls3d_unet_ASPPv2_50x25/epoch_108.pth --out out/5025/HAIS_stpls3d_unet_ASPPv2_50x25/
CUDA_VISIBLE_DEVICES=2 python test-HAIS.py configs/training_all/50x50_25/HAIS_stpls3d_unet_ATTN_ASPP_50x25.yaml pretrain/5025/HAIS_stpls3d_unet_ATTN_ASPP_50x25/epoch_108.pth --out out/5025/HAIS_stpls3d_unet_ATTN_ASPP_50x25/

echo "Rebuild HAIS 5050"
cp buff_agg/5050hierarchical_aggregation.cu lib/hais_ops/src/hierarchical_aggregation/hierarchical_aggregation.cu
cp buff_agg/5050hierarchical_aggregation.cpp lib/hais_ops/src/hierarchical_aggregation/hierarchical_aggregation.cpp
echo "Rebuild HAIS"
bash rebuild_hais.sh>log_hais_rebuild.log

echo "Semantic Segmentasi 5025"
CUDA_VISIBLE_DEVICES=2 python test-Semantic.py configs/training_only_semantic_segmentation/50x50_50/semantic_unet_50x50.yaml pretrain/5050/semantic_unet_50x50/epoch_20.pth --out out/5050/semantic_unet_50x50/ 
CUDA_VISIBLE_DEVICES=2 python test-Semantic.py configs/training_only_semantic_segmentation/50x50_50/semantic_unet_50x50_ASPPv2.yaml pretrain/5050/semantic_unet_50x50_ASPPv2/epoch_20.pth --out out/5050/semantic_unet_50x50_ASPPv2/
CUDA_VISIBLE_DEVICES=2 python test-Semantic.py configs/training_only_semantic_segmentation/50x50_50/semantic_unet_50x50_ATTN_ASPP.yaml pretrain/5050/semantic_unet_50x50_ATTN_ASPP/epoch_20.pth --out out/5050/semantic_unet_50x50_ATTN_ASPP/

echo "instance segmentation 5025"
CUDA_VISIBLE_DEVICES=2 python test-HAIS.py configs/training_all/50x50_50/HAIS_stpls3d_unet.yaml pretrain/5050/HAIS_stpls3d_unet/epoch_108.pth --out out/inst/5050/HAIS_stpls3d_unet/ 
CUDA_VISIBLE_DEVICES=2 python test-HAIS.py configs/training_all/50X50_50/HAIS_stpls3d_unet_ASPPv2.yaml pretrain/5050/HAIS_stpls3d_unet_ASPPv2/epoch_108.pth --out out/5050/HAIS_stpls3d_unet_ASPPv2/
CUDA_VISIBLE_DEVICES=2 python test-HAIS.py configs/training_all/50X50_50/HAIS_stpls3d_unet_ATTN_ASPP.yaml pretrain/5050/HAIS_stpls3d_unet_ATTN_ASPP/epoch_108.pth --out out/5050/HAIS_stpls3d_unet_ATTN_ASPP/

echo "Rebuild HAIS 10050"
cp buff_agg/10050hierarchical_aggregation.cu lib/hais_ops/src/hierarchical_aggregation/hierarchical_aggregation.cu
cp buff_agg/10050hierarchical_aggregation.cpp lib/hais_ops/src/hierarchical_aggregation/hierarchical_aggregation.cpp
bash rebuild_hais.sh>log_hais_rebuild.log
echo "Semantic Segmentasi 10050"
CUDA_VISIBLE_DEVICES=2 python test-HAIS.py configs/training_only_semantic_segmentation/100x100_50/semantic_unet_100x50.yaml pretrain/10050/semantic_unet_100x50/epoch_20.pth --out out/10050/semantic_unet_100x50/
echo "instance Segmentasi 10050"
CUDA_VISIBLE_DEVICES=2 python test-HAIS.py configs/training_all/100x100_50/HAIS_stpls3d_unet_100x50.yaml pretrain/10050/HAIS_stpls3d_unet_100x50/epoch_108.pth --out out/10050/HAIS_stpls3d_unet_100x50/

echo "Rebuild HAIS 100100"
cp buff_agg/100100hierarchical_aggregation.cu lib/hais_ops/src/hierarchical_aggregation/hierarchical_aggregation.cu
cp buff_agg/100100hierarchical_aggregation.cpp lib/hais_ops/src/hierarchical_aggregation/hierarchical_aggregation.cpp
bash rebuild_hais.sh>log_hais_rebuild.log
echo "Semantic Segmentasi 100100"
CUDA_VISIBLE_DEVICES=2 python test-HAIS.py configs/training_only_semantic_segmentation/100x100/semantic_unet_100x50.yaml pretrain/100100/semantic__unet_100x100/epoch_20.pth --out out/100100/semantic_unet_100x100/
echo "instance Segmentasi 100100"
CUDA_VISIBLE_DEVICES=2 python test-HAIS.py configs/training_all/100x100_100/HAIS_stpls3d_unet_100x100.yaml pretrain/100100/HAIS_stpls3d_unet_100x100/epoch_108.pth --out out/100100/HAIS_stpls3d_unet_100x100/

