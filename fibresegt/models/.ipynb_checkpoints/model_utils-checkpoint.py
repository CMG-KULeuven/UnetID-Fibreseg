#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Author: Rui Guo (KU Leuven), rui.guo1@kuleuven.be
Date: Jan 2023
"""

import torch
from torch import optim
from fibresegt.models.model_archs import UnetID

def get_device():
    """Returns the device to be used for training."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# def get_loss(loss_method: str):
#     """Return the loss function based on the specified loss method.

#     Args:
#         loss_method (str): Type of loss function to use. Only "BCE+dice" is avalable.

#     Returns:
#         Loss function.
#     """
#     if loss_method == "BCE+dice":
#         return tn.BCE_dice_loss
#     else:
#         raise ValueError("Invalid loss method specified. Only 'BCE+dice' is supported now.")

def get_optimizer(optimizer_method, model_params, learning_rate):
    """Return the optimizer based on the specified optimizer method.

    Args:
        optimizer_method (str): Type of optimizer to use. Only "Adam" is supported now.
        model_params (List): Model parameters to optimize.
        learning_rate (float): Learning rate for the optimizer.

    Returns:
        Optimizer.
    """
    if optimizer_method == "Adam":
        return optim.Adam(model_params, lr=learning_rate)
    else:
        raise ValueError("Invalid optimizer method specified. Only 'Adam' is supported now.")

def get_net(net_var):
    """Create a neural network for segmentation.

    Args:
        net_var (str): Type of neural network to use.

    Returns:
        The created neural network.
    """
    if net_var == 'UnetID':
        net = UnetID()
    else:
        raise ValueError(f'Invalid neural network type: {net_var}, \
                         only UnetID is supported currently.')

    return net

def load_checkpoint(checkpoint_path, model, optimizer):
    """Loads a model checkpoint and returns the model.
    
    Args:
        checkpoint_path (str): Path to the checkpoint file.
        model (torch.nn.Module): PyTorch model to load the checkpoint into.
        
    Returns:
        the trained model.
    """
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

def save_checkpoint(checkpoint_path, model):
    """Saves a model checkpoint.
    
    Args:
        checkpoint_path (str): Path to save the checkpoint file.
        model (torch.nn.Module): PyTorch model to save.
    """
    torch.save(model.state_dict(), checkpoint_path)