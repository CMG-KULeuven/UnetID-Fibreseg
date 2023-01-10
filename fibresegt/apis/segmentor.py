#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Author: Rui Guo (KU Leuven), rui.guo1@kuleuven.be
Date: July 2022
"""

import torch
import os
import time
import matplotlib.pyplot as plt
import fibresegt as fs
from torch.utils.data import DataLoader

def segm_2d_data(dataset, 
                net_var, 
                output_dir,
                trainedNet_dir,
                dataset_name='Default',
                checkpoint_id='last_id', 
                crop_slice_shape=(64,64,1), 
                constant_value=0, 
                out_threshold=0.5,
                save_orig_results=True,
                hardware_acltr='GPU',
                load_checkpoint=True,
                **postproc_param):
    """Segment a 2D dataset using a trained neural network.

    Args:
        dataset (array): The dataset to be segmented.
        net_var (str): Neural network architecture to use, now only 'UnetID' is available.
        output_dir (str): The path to save results.
        trainedNet_dir (str): The path to the trained models.
        dataset_name (str, optional): Name of the dataset, used in saving results. Defaults to 'Default'.
        checkpoint_id (str, optional): ID of the trained model to use for segmentation, either 'last_id' or 
                an epoch number. Defaults to 'last_id'.
        crop_slice_shape (tuple, optional): Shape of image to use for segmentation. Defaults to (64,64,1).
        constant_value (int, optional): The value to pad the image. Defaults to 0.
        out_threshold (float, optional): The probability threshold to use for segmentation. Defaults to 0.5.
        save_orig_results (bool, optional): Whether to save results without the postprocessing. Defaults to True.
        hardware_acltr (str, optional): Hardware accelerator to use. Defaults to 'GPU'.

    Returns:
        array: The segmentation results
    """    
    
    if hardware_acltr == 'GPU':
        device = fs.get_device()
        print(f'You are using {device} to calculate!')
        if device.type != 'cuda':
            print('Notice: the GPU is not supported!')
    else:
        device = torch.device('cpu')
        print('You have to use CPU to calculate')

    # Load the neural network
    net = fs.get_net(net_var).to(device)
    
    # checkpoint
    if checkpoint_id == 'last_id':
        checkpoint_files = os.listdir(trainedNet_dir)
        checkpoint_file = fs.join(trainedNet_dir, checkpoint_files[-1])
        net = fs.load_checkpoint(checkpoint_file, net)
    else: 
        if load_checkpoint:
            checkpoint_file = (trainedNet_dir+ '/checkpoint_epoch' 
                                +str(checkpoint_id)+'.pth')
            net = fs.load_checkpoint(checkpoint_file, net)
    net.eval()
    
    # sample
    if dataset_name == 'Default':
        segm_dir = fs.join(output_dir, 'segm_results_2D/')
        fs.mkdir(segm_dir)
    else: 
        segm_dir = fs.join(output_dir, 'segm_results_2D/'+dataset_name[:-4])
        fs.mkdir(segm_dir)

    dataset = torch.from_numpy(dataset).to(device=device, dtype=torch.float32)
    start_t = time.time()
    segm_img = fs.segm(dataset=dataset, net=net, device=device, 
                                  crop_slice_shape=crop_slice_shape, 
                                  constant_value=constant_value,
                                  out_threshold=out_threshold)
    end_t = time.time()
    print('Sementation is finished, ')
    print('The time needed is: ', end_t-start_t)
    print('Now we are saving the results ...')
    if save_orig_results or not postproc_param:
        plt.imsave(fs.join(segm_dir, 'whole_fibre.png'), segm_img[..., 1], cmap='gray')
        plt.imsave(fs.join(segm_dir, 'inner_fibre.png'), segm_img[..., 0], cmap='gray')
        plt.imsave(fs.join(segm_dir, 'fibre_edge.png'), segm_img[..., 2], cmap='gray')
    
    if postproc_param:
        method, kernel, iteration = (postproc_param['method'], 
                                     postproc_param['kernel'],  
                                     postproc_param['iteration'])
        postproc_segm_img = fs.apply_morphological_operation(data=segm_img, method=method, 
                                                            kernel=kernel, iteration=iteration)
        if postproc_param['save_postproc_results']:
            postproc_segm_dir = fs.join(segm_dir, 'postproc')
            fs.mkdir(postproc_segm_dir)
            plt.imsave(fs.join(postproc_segm_dir, 'whole_fibre.png'), 
                        postproc_segm_img[..., 1], cmap='gray')
            plt.imsave(fs.join(postproc_segm_dir, 'inner_fibre.png'), 
                        postproc_segm_img[..., 0], cmap='gray')
            plt.imsave(fs.join(postproc_segm_dir, 'fibre_edge.png'), 
                        postproc_segm_img[..., 2], cmap='gray')
    
    print('All finished!')
    return segm_img

def segm_3d_data(dataset, 
                net_var, 
                output_dir,
                trainedNet_dir,
                dataset_name='Default',
                checkpoint_id='last_id', 
                crop_slice_shape=(64,64,1), 
                constant_value=255, 
                out_threshold=0.5,
                batch_size=100,
                save_format = ['png'],
                save_image_num = 'Full',
                save_info = 'inner',
                save_orig_results=True,
                hardware_acltr='GPU',
                load_checkpoint=True,
                **postproc_param):
    """Segment a 3D dataset using a trained neural network.

    Args:
        dataset (array): The dataset to be segmented.
        net_var (str): Neural network architecture to use, now only 'UnetID' is available.
        output_dir (str): The path to save results.
        trainedNet_dir (str): The path to the trained models.
        dataset_name (str, optional): Name of the dataset, used in saving results. Defaults to 'Default'.
        checkpoint_id (str, optional): ID of the trained model to use for segmentation, either 'last_id' or 
        an epoch number. Defaults to 'last_id'.
        crop_slice_shape (tuple, optional): Shape of image to use for segmentation. Defaults to (64,64,1).
        constant_value (int, optional): The value to pad the image. Defaults to 0.
        out_threshold (float, optional): The probability threshold to use for segmentation. Defaults to 0.5.
        batch_size (int, optional): The size to batch multiple slices for segmentation at the same time. Defaults to 100.
        save_format (list, optional): The list about the format to save the resutls. Defaults to ['png'].
        save_image_num (str, optional): The number to save images for the PNG format. Defaults to 'Full'.
        save_info (str, optional): Whether to save images with full information (default: inner, edge, whole) or 
        only for inner fibre.. Defaults to 'inner'.
        save_orig_results (bool, optional): Whether to save results without the postprocessing. Defaults to True.
        hardware_acltr (str, optional): Hardware accelerator to use. Defaults to 'GPU'.
        load_checkpoint (bool, optional): Whether to load a trained model for segmentation. Defaults to True.
    """ 
  
    if hardware_acltr == 'GPU':
        device = fs.get_device()
        print(f'You are using {device} to calculate!')
        if device.type != 'cuda':
            print('Notice: the GPU is not supported!')
    else:
        device = torch.device('cpu')
        print('You have to use CPU to calculate')

    # Load the neural network
    net = fs.get_net(net_var).to(device)
    
    # checkpoint
    if checkpoint_id == 'last_id':
        checkpoint_files = os.listdir(trainedNet_dir)
        checkpoint_file = fs.join(trainedNet_dir, checkpoint_files[-1])
        net = fs.load_checkpoint(checkpoint_file, net)
    else: 
        if load_checkpoint:
            checkpoint_file = (trainedNet_dir+ '/checkpoint_epoch' 
                                +str(checkpoint_id)+'.pth')
            net = fs.load_checkpoint(checkpoint_file, net)
    net.eval()
    
    # sample
    if dataset_name == 'Default':
        segm_dir = fs.join(output_dir, 'segm_results_3D/')
        fs.mkdir(segm_dir)
    else: 
        segm_dir = fs.join(output_dir, 'segm_results_3D/'+dataset_name[:-4])
        fs.mkdir(segm_dir)

    count = 0
    ordinal = lambda n: "%d%s" % (n,"tsnrhtdd"[(n//10%10!=1)*(n%10<4)*n%10::4])
    for key, data3D in dataset.items():
        print(f'Start to deal with the {ordinal(count+1)} set of data: {key}!')
        
        data_num = data3D.shape[0]
        # Reduce the batch size to fit the GPU memory
        if data_num < 500:
            batch_data3D_load = DataLoader(data3D, shuffle=False, batch_size=int(data_num))
        else:
            batch_data3D_load = DataLoader(data3D, shuffle=False, batch_size=batch_size)
        start_t = time.time()
        segm_3Dimg_list=[fs.segm(dataset=batch_data3D, net=net, device=device, 
                            crop_slice_shape=crop_slice_shape, 
                            constant_value=constant_value,
                            out_threshold=out_threshold) for batch_data3D in batch_data3D_load]
        end_t = time.time()
        print('Sementation is finished, ')
        print('The time needed is: ', end_t-start_t)
        print('Now we are saving the results ...')
        
        if save_orig_results or not postproc_param:
            fs.save_3d_dataset([segm_dir, key], data3D=segm_3Dimg_list, 
                               save_format=save_format, 
                               save_image_num=save_image_num, 
                               save_info=save_info)
            print(f'{" "*4}Finished!')
            count += 1
        
        if postproc_param:
            method, kernel, iteration = (postproc_param['method'], 
                                         postproc_param['kernel'], 
                                         postproc_param['iteration'])
            postproc_segm_img = fs.apply_morphological_operation(data=segm_3Dimg_list, 
                                                                 method=method, kernel=kernel, 
                                                                 iteration=iteration)
            if postproc_param['save_postproc_results']:
                fs.save_3d_dataset([segm_dir, (key+'/postproc')], 
                                   data3D=postproc_segm_img, 
                                   save_format=save_format, 
                                   save_image_num=save_image_num, 
                                   save_info=save_info)
    print('All calculation is finished!')
