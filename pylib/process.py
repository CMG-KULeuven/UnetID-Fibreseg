#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Author: Rui Guo (KU Leuven), rui.guo1@kuleuven.be
Date: July 2022
"""

import numpy as np
import imlib as im
from patchify import patchify
import cv2 as cv

def patchify_images(data, shape, step):
    data_patch = patchify(data, shape, step) # step=64 for 64 steps means no overlap
    data_patches = []
    for i in range(data_patch.shape[0]):
        for j in range(data_patch.shape[1]):
            single_patch_img = data_patch[i,j,:,:]
            data_patches.append(single_patch_img)
    return np.array(data_patches)

def assemble_img(img_1, img_2, img_3):
    batch, height, width = np.array(img_1).shape
    dst_img = np.zeros((batch, height, width, 3))
    dst_img[:,:,:,0] = img_1
    dst_img[:,:,:,1] = img_2
    dst_img[:,:,:,2] = img_3
    return dst_img

def enlarge_fibre(img, kernel='cross', itera=1):
    each_enlarge_fibres_img = {}
    # Enlarge each fibre so that we can capture the out contours of each fibre
    if kernel == 'cross':
        kernel = np.array([[0,1,0],[1,1,1],[0,1,0]],np.uint8)
    elif kernel == 'full':
        kernel = np.array([[1,1,1],[1,1,1],[1,1,1]],np.uint8)
    else:
        print("There are only two options for kernel: 'corss' or 'full'")
    enlarge_fiber = cv.dilate(img,kernel,iterations = itera)
        
    return enlarge_fiber

def shrink_fibre(img, kernel='cross', itera=1):
    each_enlarge_fibres_img = {}
    count = 0
    # shrink each fibre so that we can capture the out contours of each fibre
    if kernel == 'cross':
        kernel = np.array([[0,1,0],[1,1,1],[0,1,0]],np.uint8)
    elif kernel == 'full':
        kernel = np.array([[1,1,1],[1,1,1],[1,1,1]],np.uint8)
    else:
        print("There are only two options for kernel: 'corss' or 'full'")
    enlarge_fiber = cv.erode(img, kernel,iterations = itera)
        
    return enlarge_fiber

def split_img(data, split_ratio=0.8):
    origData, label = data
    # Notice: we only slip images along the longest side of the image
    height, width = origData.shape
    if height > width:
        height_split = int(height * split_ratio)
        train_origDataLabel = [origData[0:height_split, :], label[0:height_split, :]]
        test_origDataLabel = [origData[height_split:, :], label[height_split:, :]]
    else:
        width_split = int(width * split_ratio)
        train_origDataLabel = [origData[:, 0:width_split], label[:, 0:width_split]]
        test_origDataLabel = [origData[:, width_split:], label[:, width_split:]]
    return train_origDataLabel, test_origDataLabel

def sample(data, stride_step=8, data_shape=[64, 64], shrink_size: int=0, show_img=False):
    origData, labelInnerFibre = data
    # Preprocess the origin grayscale image
    origData_patch = patchify_images(origData, shape=data_shape, step=stride_step)
    origData_patch_RGB = im.conv_3Channel(origData_patch[:,:,:,np.newaxis])
    # Preprocess the label
    # Find the edge
    if shrink_size > 0:
        labelFullFibre = enlarge_fibre(labelInnerFibre, itera=1)
        labelInnerFibre = shrink_fibre(labelInnerFibre, itera=shrink_size)
        labelFibreEdge = labelFullFibre - labelInnerFibre
    else:
        labelFullFibre = enlarge_fibre(labelInnerFibre, itera=1)
        labelFibreEdge = labelFullFibre - labelInnerFibre
    # Patchify the image
    labelInnerFibre_patch = patchify_images(labelInnerFibre, shape=data_shape, step=stride_step)
    labelFullFibre_patch = patchify_images(labelFullFibre, shape=data_shape, step=stride_step)
    labelFibreEdge_patch = patchify_images(labelFibreEdge, shape=data_shape, step=stride_step)
    # Third, Assemble these images into 3 channels
    label_patch_RGB = assemble_img(labelInnerFibre_patch, labelFullFibre_patch, labelFibreEdge_patch)
    label_patch_RGB = im.conv_3Channel(label_patch_RGB)
    image_shape = origData_patch_RGB.shape
    print(f'The number of images is {image_shape[0]} \n'
          f'The size of images is {image_shape[1:3]}')
    if show_img:
        print('Original grayscale images') 
        im.plot_multi_imgs(1, 3, origData_patch_RGB[:,:,:,:]/255)
        print('Masks for fibres without edges')
        im.plot_multi_imgs(1, 3, label_patch_RGB[:,:,:,0])
        print('Masks for whole fibres')
        im.plot_multi_imgs(1, 3, label_patch_RGB[:,:,:,1])
        print('Masks for fibre edges')
        im.plot_multi_imgs(1, 3, label_patch_RGB[:,:,:,2])
    return (origData_patch_RGB, label_patch_RGB)

def stitch_img(segm_img, step, half_step, small_patch_pred, i, j):
    if i == 0 and j ==0:
        segm_img[:, :, i:i+step+half_step, j:j+step+half_step] += small_patch_pred[:, :, 0:step+half_step, 0:step+half_step]
    elif i == 0 and j != 0:
        segm_img[:, :, i:i+step+half_step, j+half_step:j+step+half_step] += small_patch_pred[:, :, 0:step+half_step, half_step:step+half_step]
    elif i != 0 and j == 0:
        segm_img[:, :, i+half_step:i+step+half_step, j:j+step+half_step] += small_patch_pred[:, :, half_step:step+half_step, 0:step+half_step]
    else:
        segm_img[:, :, i+half_step:i+step+half_step, j+half_step:j+step+half_step] += small_patch_pred[:, :, half_step:step+half_step, half_step:step+half_step]
    return segm_img 

def sample_multiple_slices(multiple_data, multiple_labels, stride_step, data_shape, shrink_size, show_img=False):
    """Sample small images fro multiple slices

    Args:
        multiple_data (list): this list is used to save the original images data
        multiple_labels (list): this list is used to save the relevant labels
    return: the sampled images
    """
    try:
        data_num = len(multiple_data)
        label_num = len(multiple_labels)
        if data_num != label_num:
            print(f'There are {data_num} slices(images), but {label_list} labels')
    except:
        print('The type of input data and labels is not a list!')
    ordinal = lambda n: "%d%s" % (n,"tsnrhtdd"[(n//10%10!=1)*(n%10<4)*n%10::4])
    images_list = []
    masks_list = []
    for i in range(data_num):
        data, label = multiple_data[i], multiple_labels[i]
        data_size, label_size = data.shape[0:2], label.shape[0:2]
        if data_size != label_size:
            print(f'''The shape of {ordinal(i+1)} slice (image) is {data_size},
                    but the shape of relevant label is {label_size})''')
        else:
            print(f'The information for the {ordinal(i+1)} slice')
            images, masks = sample([data, label], stride_step=stride_step, data_shape=data_shape, shrink_size=shrink_size, show_img=show_img)
            images_list.append(images)
            masks_list.append(masks)
    images_patch = np.concatenate(images_list)
    masks_patch = np.concatenate(masks_list)
    print(f'The total numble of data are: {images_patch.shape[0]}')
    return (images_patch, masks_patch)