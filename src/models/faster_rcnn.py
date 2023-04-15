# -*- coding: utf-8 -*-
"""
@author: gehipeng @ 20230415
@file: faster_rcnn.py
@brief: faster_rcnn
"""
# this is a src file for defining faster-rcnn model
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class FasterRCNN(nn.Module):
    def __init__(self, num_classes, pretrained=False):
        super(FasterRCNN, self).__init__()
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.backbone = models.resnet18(pretrained=self.pretrained)
        
