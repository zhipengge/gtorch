# -*- coding: utf-8 -*-
"""
@author: gehipeng @ 20230629
@file: attention_block.py
@brief: attention_block
"""

import torch
import torch.nn as nn
import math

class SEBlock(nn.Module):
    def __init__(self, channel, divisor=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Conv2d(channel, channel // divisor, 1, 1, 0, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // divisor, channel, 1, 1, 0, bias=False),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.fc(y)
        return x * y
    
class ChannelAttention(nn.Module):
    def __init__(self, channel, divisor=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(channel, channel // divisor, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(channel // divisor, channel, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class CBAMBlock(nn.Module):
    def __init__(self, channel, divisor=8, kernel_size=7):
        super(CBAMBlock, self).__init__()
        self.channelattention = ChannelAttention(channel, divisor=divisor)
        self.spatialattention = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        x = x * self.channelattention(x)
        x = x * self.spatialattention(x)
        return x
    

class SelfAttention(nn.Module):
    def __init__(self):
        super(SelfAttention, self).__init__()
        



    

if __name__ == "__main__":
    x = torch.rand(2, 16, 32, 32)
    se_block = SEBlock(16)
    y1 = se_block(x)
    print(y1.shape)
    cbam_block = CBAMBlock(16)
    y2 = cbam_block(x)
    print(y2.shape)

