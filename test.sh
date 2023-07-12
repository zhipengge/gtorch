#!/bin/bash
source activate gtorch
num_gpus=2
checkpoint=output/checkpoints/17flowers_resnet18_20230711085847/model_64.ckpt
input="example/images/image_1330.jpg example/images/image_1357.jpg example/images/image_1358.jpg"
CUDA_VISIBLE_DEVICES=0,1 python main.py --config_path config/17flowers_resnet18.py \
    --test \
    --checkpoint ${checkpoint} \
    --input ${input}