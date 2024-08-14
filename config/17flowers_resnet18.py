# -*- coding: utf-8 -*-
"""
@author: gehipeng @ 20230706
@file: 17flowers_resnet18.py
@brief: 17flowers_resnet18
"""
from config.base import BaseConfig

config = BaseConfig(
    BATCH_SIZE=256,
    NUM_WORKERS=8,
    NUM_EPOCHS=70,
    CHECKPOINT_DIR="output/checkpoints",
    DATA_DIR="/home/gezhipeng/mnt1/dataset/17flowers",
    GPUS="0",
    MODEL_CONFIG={
        "name": "resnet18",
        "weights": "IMAGENET1K_V1",
        "pretrained": True,
        "num_classes": 17,
    },
    INPUT_SIZE=(224, 224),
    NUM_STEPS_TO_DISPLAY=10,
    RESUME=True,
    TRANSFORMS={
        "train": [
            {
                "name": "RandomResizedCrop",
                "args": {
                    "size": (224, 224),
                    "scale": (0.08, 1.0),
                    "ratio": (0.75, 1.3333333333333333),
                    # 'interpolation': 2
                },
            },
            {"name": "RandomHorizontalFlip", "args": {"p": 0.5}},
            {"name": "RandomVerticalFlip", "args": {"p": 0.5}},
            {
                "name": "RandomRotation",
                "args": {
                    "degrees": 15,
                    # 'resample': False,
                    "expand": False,
                    "center": None,
                    "fill": 0,
                },
            },
            {
                "name": "ColorJitter",
                "args": {
                    "brightness": 0.4,
                    "contrast": 0.4,
                    "saturation": 0.4,
                    "hue": 0.4,
                },
            },
            {"name": "ToTensor", "args": {}},
            {
                "name": "Normalize",
                "args": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
            },
        ],
        "val": [
            {
                "name": "Resize",
                "args": {
                    "size": (224, 224),
                    # 'interpolation': 2
                },
            },
            {"name": "ToTensor", "args": {}},
            {
                "name": "Normalize",
                "args": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
            },
        ],
    },
)
