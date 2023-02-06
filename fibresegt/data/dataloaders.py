#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Author: Rui Guo (KU Leuven), rui.guo1@kuleuven.be
Date: 2022
"""

import torch
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from fibresegt.data.datasets import CTImagesDataset

def get_dataloaders(data, val_percent, batch_size, data_aug=None):
    """
    Generate train and validation dataloaders from the given data.
    
    Args:
        data (tuple): images and masks
        val_percent (float): Percent of dataset to use for validation.
        batch_size (int): Number of images in each batch.
        data_aug (dict, optional): Dictionary of data augmentation parameters. Defaults to None.
        
    Returns:
        tuple: Tuple of dataloaders and dataset number for training and validation.
    """
    if data_aug is not None:
        brightness_value = data_aug['brightness']
        contrast_value = data_aug['contrast']
        GaussianBlur_kernel = data_aug['GaussianBlur_kernel']
        GaussianBlur_sigma = data_aug['GaussianBlur_sigma']
        data_transform = transforms.Compose([transforms.ColorJitter(brightness=brightness_value, 
                                                                    contrast=contrast_value),
                                            transforms.GaussianBlur(kernel_size=GaussianBlur_kernel, 
                                                                sigma=GaussianBlur_sigma),
                                            transforms.ToTensor()])
    else:
        data_transform = transforms.ToTensor()
    dataset = CTImagesDataset(data, transform=data_transform)
    
    # Split into train / validation partitions
    num_val = int(len(dataset) * val_percent)
    num_train = len(dataset) - num_val
    train_set, val_set = random_split(dataset, [num_train, num_val], generator=torch.Generator().manual_seed(0))
    
    # Check the data values
    max_value = torch.max(train_set[0][0])
    if max_value > 1.0:
        print(f'The maximum values of the data is {max_value}, please nomalize it between 0-1!')
        
    # Create the dataloaders
    loader_args = dict(batch_size=batch_size, num_workers=4, pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, drop_last=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)
    
    return (train_loader, num_train), (val_loader, num_val)