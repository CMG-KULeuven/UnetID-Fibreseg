#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Author: Rui Guo (KU Leuven), rui.guo1@kuleuven.be
Date: July 2022
"""

import functools
import math
import random
import numpy as np
import pylib as py
import torch
import torchvision.transforms.functional as F

class RandomRotFixAngle(torch.nn.Module):    
    """Rotate the given image randomly with a given probability.

    Args:
        fix_angle (float): the angle of the image being rotated. Default value is 90.
        p (float): probability of the image being rotated. Default value is 0.5
    """

    def __init__(self, fix_angle, p=0.5):
        super().__init__()
        self.p = p
        self.fix_angle = fix_angle

    def forward(self, img):
        """
        Args:
            img (PIL Image): Image to be flipped.

        Returns:
            PIL Image: Randomly flipped image.
        """
        if torch.rand(1) < self.p:
            return img.rotate(self.fix_angle)
        return img

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(angle={self.fix_angle}, p={self.p})"
    
class RandomFlip(torch.nn.Module):
    """Flip the given image at the horizontal or vertical direction
    randomly with a given probability.

    Args:
        flip_method: 'horizontal' or 'vertical'. Default is 'horizontal'.
                     'horizontal' means flip left and right
                     'vertical' means flip top and bottom
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, flip_method='horizontal', p=0.5):
        super().__init__()
        self.p = p
        self.flip_method = flip_method

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be flipped.

        Returns:
            PIL Image or Tensor: Randomly flipped image.
        """
        if torch.rand(1) < self.p:
            if self.flip_method == 'horizontal':
                return F.hflip(img)
            elif self.flip_method == 'vertical':
                return F.vflip(img)
            else:
                print("Please check whether the variable name of flip_method is horizontal or vertical")
        return img

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(flip_method={self.flip_method}, p={self.p})"

        