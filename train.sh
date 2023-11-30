#!/bin/bash
source activate gtorch
num_gpus=2
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=${num_gpus} main.py --config_path config/17flowers_resnet50.py 