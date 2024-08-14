# -*- coding: utf-8 -*-
"""
@author: gehipeng @ 20230415
@file: base.py
@brief: base
"""
# this is a src file for defining base config
import os


class BaseConfig:
    def __init__(self, *args, **kwargs):
        PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.BATCH_SIZE = kwargs.get("BATCH_SIZE", 128)
        self.NUM_WORKERS = kwargs.get("NUM_WORKERS", 8)
        self.NUM_EPOCHS = kwargs.get("NUM_EPOCHS", 100)
        self.CHECKPOINT_DIR = kwargs.get(
            "CHECKPOINT_DIR", os.path.join(PROJECT_DIR, "output/checkpoints")
        )
        self.DATA_DIR = kwargs.get("DATA_DIR", None)
        self.GPUS = kwargs.get("GPUS", None)
        self.MODEL_NAME = kwargs.get("MODEL_NAME", None)
        self.MODEL_CONFIG = kwargs.get("MODEL_CONFIG", None)
        self.INPUT_SIZE = kwargs.get("INPUT_SIZE", (None, None))
        self.TRANSFORMS = kwargs.get("TRANSFORMS", {})
        self.NUM_STEPS_TO_DISPLAY = kwargs.get("NUM_STEPS_TO_DISPLAY", 10)
        self.RESUME = kwargs.get("RESUME", False)
        self.SEED = kwargs.get("SEED", 42)

    def __repr__(self) -> str:
        ss = f" {self.__class__.__name__} ".center(100, "*")
        for k, v in self.__dict__.items():
            ss += f"\n> {k}: {v}"
        ss += "\n"
        ss += f" {self.__class__.__name__} ".center(100, "*")
        return ss

    def get_config(self):
        return self.__dict__


if __name__ == "__main__":
    config = BaseConfig(BATCH_SIZE=1)
    print(config)
