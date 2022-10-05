#!/usr/bin/env bash
CONFIG=$1
GPUS=$2
PORT=${PORT:-29500}

OMP_NUM_THREADS=1 torchrun --nproc_per_node=$GPUS --master_port=$PORT train-HAIS.py --dist $CONFIG ${@:3} --resume work_dirs/HAIS_stpls3d_unet_50X25/epoch_52.pth
