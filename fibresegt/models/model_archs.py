#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Refer: 
This project incorporates code from the Pytorch-UNet library (available at https://github.com/milesial/Pytorch-UNet) 
and the U-Net-id methods (described in the paper at https://www.mdpi.com/2072-4292/12/10/1544). 
The Pytorch-UNet library is licensed under the GPL-3.0 license."
"""

import torch.nn as nn
import torch

class conv_block(nn.Module):
    """
    Double Convolution Block
    """
  
    def __init__(self, in_ch=3, out_ch=1, mid_ch=None):
        super().__init__()
        
        if not mid_ch:
            mid_ch = out_ch

        self.conv1 = nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(mid_ch)

        self.conv2 = nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(out_ch)

        self.relu = nn.ReLU(inplace=True) 
    
    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        return x
        
        
class up_conv(nn.Module):
    """
    Up Convolution Block
    """
    def __init__(self, in_ch, out_ch, is_deconv=True):
        super(up_conv, self).__init__()
        
        if is_deconv:
            self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        else:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2),
                nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
                )

    def forward(self, x):
        x = self.up(x)
        return x
    
class Unet(nn.Module):
    """
    UNet Implementation
    """
    def __init__(self, in_ch=3, out_ch=1, n1 = 16):
        super(Unet, self).__init__()

        n_ch = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.Conv1 = conv_block(in_ch, n_ch[0])
        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv2 = conv_block(n_ch[0], n_ch[1])
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.Conv3 = conv_block(n_ch[1], n_ch[2])
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.Conv4 = conv_block(n_ch[2], n_ch[3])
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.Conv5 = conv_block(n_ch[3], n_ch[4])

        self.Up5 = up_conv(n_ch[4], n_ch[3])
        self.Up_conv5 = conv_block(n_ch[4], n_ch[3])

        self.Up4 = up_conv(n_ch[3], n_ch[2])
        self.Up_conv4 = conv_block(n_ch[3], n_ch[2])

        self.Up3 = up_conv(n_ch[2], n_ch[1])
        self.Up_conv3 = conv_block(n_ch[2], n_ch[1])

        self.Up2 = up_conv(n_ch[1], n_ch[0])
        self.Up_conv2 = conv_block(n_ch[1], n_ch[0])

        self.Conv = nn.Conv2d(n_ch[0], out_ch, kernel_size=1, stride=1)
        self.active = nn.Sigmoid()

    def forward(self, x):

        e1 = self.Conv1(x)

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)

        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)

        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)

        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5)

        d5 = self.Up5(e5)
        d5 = torch.cat((e4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((e3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        out = self.Conv(d2)
        out = self.active(out)

        return out
    
class cnn_part(nn.Module):
    """
    Convolution Neural Network Part
    """
    def __init__(self, in_ch=3, out_ch=1):
        super().__init__()
        
        self.ConvB1 = conv_block(in_ch, 32, 64)
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
        self.Unet1 = Unet(1, 1)
        self.Unet2 = Unet(1, 1)
        self.Unet3 = Unet(1, 1)
        
        self.CNN_part1 = cnn_part(3, 1)
        self.CNN_part2 = cnn_part(3, 1)
        self.CNN_part3 = cnn_part(3, 1)
    
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