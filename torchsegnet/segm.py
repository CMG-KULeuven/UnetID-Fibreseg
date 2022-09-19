#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Author: Rui Guo (KU Leuven), rui.guo1@kuleuven.be
Date: July 2022
"""

import numpy as np
import torchsegnet as tn
import tqdm
import torchlib as tl
import pylib as py
import imlib as im
import matplotlib.pyplot as plt 
import torch
import os
from torch.utils.data import DataLoader
import torch.nn.functional as F

def segm(dataset, 
         net, 
         device, 
         crop_slice_shape, 
         constant_value: int = 0, 
         out_threshold: float = 0.5):
	"""
	This is based on the pytorch tensor. 
	dataset: if the dataset is 2D case, then the shape is (height, width)
	         if the dataset is 3D case, then the shape is (depth, height, width)
	constant_value = 0 or 255
	crop_slice_shape: This is for size of images to be cropped
	constant_value: This is used for padding
	"""
	#This will split the image into small images of shape
	# The dimension of the original image
	step= int(crop_slice_shape[0]/2)
	half_step = int(step/2)
	ndim_dataset = dataset.ndim
	if ndim_dataset == 2:
		height, width = dataset.shape
		dataset = dataset[None,..., None]
		# if crop_slice_shape[2] == 3:
		# 	# dataset = dataset[...,None]
		# 	# dataset = im.conv_3Channel(dataset)
		# 	print('Currently, we only support the single channel')
	else:
		depth, height, width = dataset.shape
		dataset = dataset[..., None]
		# if width <= 3:
		# 	print(f'The width of the dataset is only {width}, please check whether the input dataset is 3D data!')
		# 	if crop_slice_shape[2] == 3:
		# 		# dataset = dataset[...,np.newaxis]
		# 		# dataset = im.conv_3Channel(dataset)
		# 		print('Currently, we only support the single channel')
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
				segm_img = py.stitch_img(segm_img, step, half_step, small_patch_pred, i, j)
	segm_img = segm_img.permute([0,2,3,1]).cpu().detach().numpy()
	if ndim_dataset == 2:
		return segm_img[0, 0:height, 0:width, :]
	elif ndim_dataset == 3:
		return segm_img[:, 0:height, 0:width, :]

def segm_2D(dataset, 
						net_var, 
						output_dir,
						trainedNet_dir,
						dataset_name='Default',
						checkpoint_id='last_id', 
						crop_slice_shape=(64,64,1), 
						constant_value=255, 
						out_threshold=0.5,
						save_results=True,
						hardware_acltr='GPU',
						load_checkpoint=False,
						load_trainingmodel=True,
						**postproc_param):
	# Currently there is only one way to load the pretrained model by using checkpoint.

	if hardware_acltr == 'GPU':
		device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
		if device.type != 'cuda':
			print('You have to use CPU to calculate')
	else:
		device = torch.device('cpu')

	# specify the model to use
	if load_checkpoint:
		if net_var == 'UnetID':
			net = tn.UnetID()
		elif net_var == 'Unet':
			net = tn.Unet()
		net.to(device=device)
	print(f'load_checkpoint: {load_checkpoint}')
	print(f'load_trainingmodel: {load_trainingmodel}')
	# checkpoint
	if checkpoint_id == 'last_id':
		checkpoint_files = os.listdir(trainedNet_dir)
		checkpoint_file = py.join(trainedNet_dir, checkpoint_files[-1])
		if load_checkpoint:
			checkpoint = torch.load(checkpoint_file)
			net.load_state_dict(checkpoint)
		elif load_trainingmodel:
			net = torch.jit.load(checkpoint_file)
			net.to(device=device)
	else: 
		if load_checkpoint:
			checkpoint_file = trainedNet_dir+ '/checkpoint_epoch'+str(checkpoint_id)+'.pth'
			checkpoint = torch.load(checkpoint_file)
			net.load_state_dict(checkpoint)
		elif load_trainingmodel:
			checkpoint_file = trainedNet_dir+ '/model_epoch_'+str(checkpoint_id)+'.pt'
			net = torch.jit.load(checkpoint_file)
			net.to(device=device)
	
	
	# model_scripted = torch.jit.script(net) # Export to TorchScript
	# model_scripted.save(output_dir+'model_scripted.pt') # Save

	net.eval()
	# sample
	if dataset_name == 'Default':
		segm_dir = py.join(output_dir, 'segm_results_2D/')
		py.mkdir(segm_dir)
	else: 
		segm_dir = py.join(output_dir, dataset_name[:-4])
		py.mkdir(segm_dir)

	dataset = torch.from_numpy(dataset).to(device=device, dtype=torch.float32)
	segm_img = segm(dataset=dataset, net=net, device=device, 
								  crop_slice_shape=crop_slice_shape, 
								  constant_value=constant_value,
								  out_threshold=out_threshold)

	if save_results:
		plt.imsave(py.join(segm_dir, 'whole_fibre.png'), segm_img[..., 1], cmap='gray')
		plt.imsave(py.join(segm_dir, 'inner_fibre.png'), segm_img[..., 0], cmap='gray')
		plt.imsave(py.join(segm_dir, 'fibre_edge.png'), segm_img[..., 2], cmap='gray')
	
	if postproc_param:
		method, kernel, iteration = postproc_param['method'], postproc_param['kernel'], postproc_param['iteration']
		postproc_segm_img = im.morphology(data=segm_img, method=method, kernel=kernel, iteration=iteration)
		if postproc_param['save_postproc_results']:
			postproc_segm_dir = py.join(segm_dir, 'postproc')
			py.mkdir(postproc_segm_dir)
			plt.imsave(py.join(postproc_segm_dir, 'whole_fibre.png'), postproc_segm_img[..., 1], cmap='gray')
			plt.imsave(py.join(postproc_segm_dir, 'inner_fibre.png'), postproc_segm_img[..., 0], cmap='gray')
			plt.imsave(py.join(postproc_segm_dir, 'fibre_edge.png'), postproc_segm_img[..., 2], cmap='gray')
	
	print('Calculation is finished!')
	return segm_img

def segm_3D4D(dataset, 
			  net_var, 
			  output_dir,
			  trainedNet_dir,
			  dataset_name='Default',
			  checkpoint_id='last_id', 
			  crop_slice_shape=(64,64,1), 
			  constant_value=255, 
			  out_threshold=0.5,
			  save_format = ['png'],
			  save_results=True,
			  hardware_acltr='GPU',
			  load_checkpoint=True,
			  load_trainingmodel=False,
			  **postproc_param):
	# Currently there is only one way to load the pretrained model by using checkpoint.

	if hardware_acltr == 'GPU':
		device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
		print(f'You are using {device} to calculate!')
		if device.type != 'cuda':
			print('You have to use CPU to calculate')
	else:
		device = torch.device('cpu')
		print(f'You are using {device} to calculate!')

	
	# specify the model to use
	if load_checkpoint:
		if net_var == 'UnetID':
			net = tn.UnetID()
		elif net_var == 'Unet':
			net = tn.Unet()
		net.to(device=device)
	print(f'load_checkpoint: {load_checkpoint}')
	print(f'load_trainingmodel: {load_trainingmodel}')
	# checkpoint
	if checkpoint_id == 'last_id':
		checkpoint_files = os.listdir(trainedNet_dir)
		checkpoint_file = py.join(trainedNet_dir, checkpoint_files[-1])
		if load_checkpoint:
			checkpoint = torch.load(checkpoint_file)
			net.load_state_dict(checkpoint)
		elif load_trainingmodel:
			net = torch.jit.load(checkpoint_file)
			net.to(device=device)
	else: 
		if load_checkpoint:
			checkpoint_file = trainedNet_dir+ '/checkpoint_epoch'+str(checkpoint_id)+'.pth'
			checkpoint = torch.load(checkpoint_file)
			net.load_state_dict(checkpoint)
		elif load_trainingmodel:
			checkpoint_file = trainedNet_dir+ '/model_epoch_'+str(checkpoint_id)+'.pt'
			net = torch.jit.load(checkpoint_file)
			net.to(device=device)
	net.eval()

	# sample
	if dataset_name == 'Default':
		segm_dir = py.join(output_dir, 'segm_results_3D4D/')
		py.mkdir(segm_dir)
	else: 
		segm_dir = py.join(output_dir, dataset_name[:-4])
		py.mkdir(segm_dir)

	segm_3D4Dimg = {}
	count = 0
	ordinal = lambda n: "%d%s" % (n,"tsnrhtdd"[(n//10%10!=1)*(n%10<4)*n%10::4])
	for key, data3D in dataset.items():
		print(f'Start to deal with the {ordinal(count+1)} set of data: {key}!')
		
		data_num = data3D.shape[0]
		# Reduce the batch size to fit the GPU memory
		if data_num < 100:
			batch_data3D_load = DataLoader(data3D, shuffle=False, batch_size=int(data_num))
		else:
			batch_data3D_load = DataLoader(data3D, shuffle=False, batch_size=int(data_num/5))
		segm_3Dimg_list=[segm(dataset=batch_data3D, net=net, device=device, 
													crop_slice_shape=crop_slice_shape, 
													constant_value=constant_value,
													out_threshold=out_threshold) for batch_data3D in batch_data3D_load]
						
		if save_results:
			im.save_3Ddata([segm_dir, key], data3D=segm_3Dimg_list, save_format=save_format)
			print(f'{" "*4}Finished!')
			count += 1
		
		if postproc_param:
			method, kernel, iteration = postproc_param['method'], postproc_param['kernel'], postproc_param['iteration']
			postproc_segm_img = im.morphology(data=segm_3Dimg_list, method=method, kernel=kernel, iteration=iteration)
			if postproc_param['save_postproc_results']:
				im.save_3Ddata([segm_dir, (key+'/postproc')], data3D=postproc_segm_img, save_format=save_format)
	print('All calculation is finished!')
