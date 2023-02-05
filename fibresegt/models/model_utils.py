#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from torch import optim
import torch.nn.functional as F
from fibresegt.models.model_archs import UnetID
from fibresegt.data.postproc_utils import slide_window

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

def load_checkpoint(checkpoint_file, model):
    """Loads a model checkpoint and returns the model.
    
    Args:
        checkpoint_path (str): Path to the checkpoint file.
        model (torch.nn.Module): PyTorch model to load the checkpoint into.
        
    Returns:
        the trained model.
    """
    model.load_state_dict(torch.load(checkpoint_file))
    return model

def save_checkpoint(checkpoint_path, model):
    """Saves a model checkpoint.
    
    Args:
        checkpoint_path (str): Path to save the checkpoint file.
        model (torch.nn.Module): PyTorch model to save.
    """
    torch.save(model.state_dict(), checkpoint_path)
    
def segm(dataset, 
         net, 
         device, 
         crop_slice_shape, 
         constant_value: int = 0, 
         out_threshold: float = 0.5):
	"""Segment an image using a trained neural network.

	Args:
		dataset (torch.Tensor): The image to segment.
		net (torch.nn.Module): The trained neural network to use for segmentation.
		device (torch.device): The device to run the model (CPU or GPU)
		crop_slice_shape (_type_): The shape of the image to crop and segment
		constant_value (int, optional): The value to pad the image. Defaults to 0.
		out_threshold (float, optional): The probability threshold to use for segmentation. Defaults to 0.5.

	Returns:
		array: The segmented image with shape (height, width) or (depth, height, width) depending on the input image
  
	Note for dataset:
		if the dataset is 2D case, then the shape is (height, width)
		if the dataset is 3D case, then the shape is (depth, height, width)
  
	""" 
	#This will split the image into small images of shape
	# The dimension of the original image
	step= int(crop_slice_shape[0]/2)
	half_step = int(step/2)
	ndim_dataset = dataset.ndim
	if ndim_dataset == 2:
		height, width = dataset.shape
		dataset = dataset[None,..., None]
	else:
		depth, height, width = dataset.shape
		dataset = dataset[..., None]
	dataset_padding = F.pad(dataset, 
						   (0, 0, 0, crop_slice_shape[1], 0, crop_slice_shape[0]),
							'constant',  value=constant_value)
	depth, height_padding, width_padding, _ = dataset_padding.shape
	dataset_padding = (dataset_padding/255.0).permute([0,3,1,2])
	segm_img = torch.zeros((depth, 3, height_padding, width_padding)).to(device=device, dtype=torch.uint8)

	for i in range(0, height, step):
		for j in range(0, width, step):
			small_patch = dataset_padding[:, :, i:i+step*2, j:j+step*2]
			small_patch = small_patch.to(device=device, dtype=torch.float32)
			if small_patch.shape[2] == small_patch.shape[3] and small_patch.shape[2]>=crop_slice_shape[0]:
				#Predict and threshold for values above 0.5 probability
				small_patch_pred = ((net(small_patch) >= out_threshold)*255).to(dtype=torch.uint8)
				segm_img = slide_window(segm_img, step, half_step, small_patch_pred, i, j)
	segm_img = segm_img.permute([0,2,3,1]).cpu().detach().numpy()
	if ndim_dataset == 2:
		return segm_img[0, 0:height, 0:width, :]
	elif ndim_dataset == 3:
		return segm_img[:, 0:height, 0:width, :]