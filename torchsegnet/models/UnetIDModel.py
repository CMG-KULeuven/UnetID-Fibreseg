#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Author: Rui Guo (KU Leuven), rui.guo1@kuleuven.be
Date: July 2022
Refer: 
https://www.mdpi.com/2072-4292/12/10/1544
"""

import numpy as np
import torchsegnet.models as segModel
import torch.nn as nn
import torch

class CNN_part(nn.Module):
    """
    Convolution Neural Network Part
    """
    def __init__(self, in_ch=3, out_ch=1):
        super().__init__()
        
        self.ConvB1 = segModel.conv_block(in_ch, 32, 64)
        self.Conv2 = nn.Conv2d(32, out_ch, kernel_size=3, padding=1, bias=True)
        self.active = nn.Sigmoid()

    def forward(self, x):
        x = self.ConvB1(x)
        x = self.Conv2(x)
        x = self.active(x)
        return x

class UnetID(nn.Module):
    """
    Unet_ID Implementation
    """
    def __init__(self):
        super(UnetID, self).__init__()
        self.Unet1 = segModel.Unet(1, 1)
        self.Unet2 = segModel.Unet(1, 1)
        self.Unet3 = segModel.Unet(1, 1)
        
        self.CNN_part1 = CNN_part(3, 1)
        self.CNN_part2 = CNN_part(3, 1)
        self.CNN_part3 = CNN_part(3, 1)
    
    def forward(self, x):
        u1 = self.Unet1(x)
        u2 = self.Unet2(x)
        u3 = self.Unet3(x)
        u4 = torch.cat((u1, u2, u3), dim=1)
        c1 = self.CNN_part1(u4)
        c2 = self.CNN_part2(u4)
        c3 = self.CNN_part3(u4)
        out = torch.cat((c1, c2, c3), dim=1)
        return out