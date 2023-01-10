#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Author: Rui Guo (KU Leuven), rui.guo1@kuleuven.be
Date: July 2022
"""

import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import random

class CTImagesDataset(Dataset):
    def __init__(self, data, transform=None):
      # 1. Initialize file paths or a list of file names.
        self.images = data[0]
        self.masks = data[1]
        self.transform = transform
        # assert self.images.shape == self.masks.shape, \
        # f'Images and masks should have the same shape, but are {images.shape} and {masks.shape}' 
      
    def __len__(self):
        return len(self.images)
      
    def __getitem__(self, index):
        image = self.images[index]
        mask = self.masks[index]
        image = Image.fromarray(image.astype(np.uint8))
        mask = Image.fromarray(mask.astype(np.uint8))
        # Preprocess and augment data.
        # make a seed with numpy generator 
        seed = np.random.randint(2147483647)
        if self.transform is not None:
            random.seed(seed) # apply this seed to img transforms
            torch.manual_seed(seed)
            image = self.transform(image)
            random.seed(seed) # apply this seed to target transforms
            torch.manual_seed(seed)
            mask = self.transform(mask)
        return image, mask