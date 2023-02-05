#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Refer: 
This project uses codes are from the Pytorch-UNet library (available at https://github.com/milesial/Pytorch-UNet/) 
The Pytorch-UNet library is licensed under the GPL-3.0 license."
"""

import torch
from tqdm import tqdm
from fibresegt.models.loss import *

def evaluate_net(net, dataloader, device):
    """Evaluate the performance of a segmentation model on a dataset.

    Args:
        net: The model to evaluate.
        dataloader: The dataloader to use for evaluation.
        device: The device to use for computation.

    Returns:
        tuple: The average Dice score and average loss over the validation set.
    """    
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0
    loss = 0

    # iterate over the validation set
    for val_images, val_masks in tqdm(dataloader, total=num_val_batches, desc='Validation', unit='batch', leave=False):
        # move images and labels to correct device and type
        val_images = val_images[:,0:1,:,:].to(device=device, dtype=torch.float32)
        val_masks = val_masks.to(device=device, dtype=torch.float32)

        with torch.no_grad():
            # predict the mask
            pred_mask = net(val_images)
            
             # compute the Dice score
            dice_score += dice_coeff(pred_mask, val_masks, reduce_batch_first=False)
            # compute the loss
            loss += (BCE_loss(pred_mask, val_masks) + dice_loss(pred_mask, val_masks))

    net.train()

    # Fixes a potential division by zero error
    if num_val_batches == 0:
        return dice_score, loss
    return (dice_score / num_val_batches), (loss/num_val_batches)
