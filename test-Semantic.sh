CUDA_VISIBLE_DEVICES=4 python test-HAIS.py configs/training_only_semantic_segmentation/50x50_50/semantic_unet_50x50.yaml pretrain/5050/semantic_unet_50x50/epoch_20.pth --out out/5050/semantic_unet_50x50/ 
CUDA_VISIBLE_DEVICES=4 python test-HAIS.py configs/training_only_semantic_segmentation/50x50_50/semantic_unet_50x50_ASPP.yaml pretrain/5050/semantic_unet_50x50_ASPP/epoch_20.pth --out out/5050/semantic_unet_50x50_ASPP/
CUDA_VISIBLE_DEVICES=4 python test-HAIS.py configs/training_only_semantic_segmentation/50x50_50/semantic_unet_50x50_ASPPv2.yaml pretrain/5050/semantic_unet_50x50_ASPPv2/epoch_20.pth --out out/5050/semantic_unet_50x50_ASPPv2/
CUDA_VISIBLE_DEVICES=4 python test-HAIS.py configs/training_only_semantic_segmentation/50x50_50/semantic_unet_50x50_ATTN_ASPP.yaml pretrain/5050/semantic_unet_50x50_ATTN_ASPP/epoch_20.pth --out out/5050/semantic_unet_50x50_ATTN_ASPP/


CUDA_VISIBLE_DEVICES=4 python test-HAIS.py configs/training_only_semantic_segmentation/50x50_25/semantic_unet_50x25.yaml pretrain/5025/semantic_unet_50x25/epoch_20.pth --out out/5025/semantic_unet_50x25/ 
CUDA_VISIBLE_DEVICES=4 python test-HAIS.py configs/training_only_semantic_segmentation/50x50_25/semantic_unet_50x25_ASPP.yaml pretrain/5025/semantic_unet_50x25_ASPP/epoch_16.pth --out out/5025/semantic_unet_50x25_ASPP/
CUDA_VISIBLE_DEVICES=4 python test-HAIS.py configs/training_only_semantic_segmentation/50x50_25/semantic_unet_50x25_ASPPv2.yaml pretrain/5025/semantic_unet_50x25_ASPPv2/epoch_16.pth --out out/5025/semantic_unet_50x25_ASPPv2/
CUDA_VISIBLE_DEVICES=4 python test-HAIS.py configs/training_only_semantic_segmentation/50x50_25/semantic_unet_50x25_ATTN_ASPP.yaml pretrain/5025/semantic_unet_50x25_ATTN_ASPP/epoch_16.pth --out out/5025/semantic_unet_50x25_ATTN_ASPP/
