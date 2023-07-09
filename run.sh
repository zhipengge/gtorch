#!/bin/bash
source activate gtorch
num_gpus=2
python -m torch.distributed.launch --nproc_per_node=${num_gpus} main.py --config_path config/17flowers_resnet18.py