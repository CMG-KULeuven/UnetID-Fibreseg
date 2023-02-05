#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from patchify import patchify
import cv2 as cv
from fibresegt.data.data_utils import conv_rgb_img, remove_border_fibre
from fibresegt.vis.plot_utils import plot_imgs

def create_image_patches(data, shape, step):
    """Generate patches for a large 2D image size

    Args:
        data (array): The image to be patched.
        shape (list): The shape (height, width) of the patches to be generated.
        step (int): The step size for patch sampling.

    Returns:
        array: An array of patches extracted from the input image.
    """    
    data_patch = patchify(data, shape, step) # step=64 for 64 steps means no overlap
    data_patches = []
    for i in range(data_patch.shape[0]):
        for j in range(data_patch.shape[1]):
            single_patch_img = data_patch[i,j,:,:]
            data_patches.append(single_patch_img)
    return np.array(data_patches)

def assemble_img(img_1, img_2, img_3):
    # Assemble three images into a single image.
    batch, height, width = np.array(img_1).shape
    dst_img = np.zeros((batch, height, width, 3))
    dst_img[:,:,:,0] = img_1
    dst_img[:,:,:,1] = img_2
    dst_img[:,:,:,2] = img_3
    return dst_img

def enlarge_fibre(img, kernel='cross', itera=1):
    """Enlarge fibres in an image using morphological dilation.

    Args:
        img (array): The input image.
        kernel (str, optional): The shape of the morphological kernel to use. Can be 'cross' or 'full'. Defaults to 'cross'.
        itera (int, optional): The number of dilation iterations to perform. Defaults to 1.
        
    Returns:
        array: The enlarged fibre image.
    """    
    if kernel == 'cross':
        kernel = np.array([[0,1,0],[1,1,1],[0,1,0]],np.uint8)
    elif kernel == 'full':
        kernel = np.array([[1,1,1],[1,1,1],[1,1,1]],np.uint8)
    else:
        print("There are only two options for kernel: 'corss' or 'full'")
    enlarged_fibre = cv.dilate(img,kernel,iterations = itera)
        
    return enlarged_fibre

def shrink_fibre(img, kernel='cross', itera=1):
    """Shrink fibres in the image using morphological erosion.

    Args:
        img (array): The input image._
        kernel (str, optional): The shape of the morphological kernel to use. Can be 'cross' or 'full'. Defaults to 'cross'.
        itera (int, optional): The number of erosion iterations to perform. Defaults to 1.

    Returns:
        array: The shrinked fibre image.
    """    
    # shrink each fibre so that we can capture the out contours of each fibre
    if kernel == 'cross':
        kernel = np.array([[0,1,0],[1,1,1],[0,1,0]],np.uint8)
    elif kernel == 'full':
        kernel = np.array([[1,1,1],[1,1,1],[1,1,1]],np.uint8)
    else:
        print("There are only two options for kernel: 'corss' or 'full'")
    shrinked_fibre = cv.erode(img, kernel,iterations = itera)
        
    return shrinked_fibre

def generate_training_samples(data, stride_step=8, data_shape=[64, 64], itera_shrink_size: int=0, 
                              itera_enlarge_size: int=2, remove_border_fibres=False, 
                              show_img=False, fig_path=None, save_fig=False):
    """Sample CT images and masks.

    Args:
        data (tuple): CT images and masks
        stride_step (int, optional): step size for extracting patches. Defaults to 8.
        data_shape (list, optional): shape of patches. Defaults to [64, 64].
        itera_shrink_size (int, optional): iteration to erode inner fibres. Defaults to 0.
        itera_enlarge_size (int, optional): iteration to dilate inner fibres. Defaults to 2.
        remove_border_fibres (bool, optional): whether to remove fibres at the border of the images. Defaults to False.
        show_img (bool, optional): whether to display the images. Defaults to False.
        fig_path (_type_, optional): file path to save figures. Defaults to None.
        save_fig (bool, optional): whether to save figures. Defaults to False.

    Returns:
        tuple: the sampled images for the images and labels
    """    
    origData, labelInnerFibre = data
    # Preprocess the origin grayscale image
    origData_patch = create_image_patches(origData, shape=data_shape, step=stride_step)
    labelInnerFibre_patch = create_image_patches(labelInnerFibre, shape=data_shape, step=stride_step)
    if remove_border_fibres:
        labelInnerFibre_patch = [remove_border_fibre(image) for image in labelInnerFibre_patch]
        print('remove_border_fibres')
        
    origData_patch_RGB = conv_rgb_img(origData_patch[:,:,:,np.newaxis])
    # Preprocess the label
    # Find the edge
    if itera_shrink_size > 0:
        labelFullFibre_patch = [enlarge_fibre(labelInnerFibre, itera=itera_enlarge_size) for labelInnerFibre in labelInnerFibre_patch]
        labelInnerFibre_patch = [shrink_fibre(labelInnerFibre, itera=shritera_shrink_sizeink_size) for labelInnerFibre in labelInnerFibre_patch]
        labelFibreEdge_patch = np.array(labelFullFibre_patch) - np.array(labelInnerFibre_patch)*1
    else:
        labelFullFibre_patch = [enlarge_fibre(labelInnerFibre, itera=itera_enlarge_size) for labelInnerFibre in labelInnerFibre_patch]
        labelFibreEdge_patch = np.array(labelFullFibre_patch) - np.array(labelInnerFibre_patch) *1
    # Third, Assemble these images into 3 channels
    label_patch_RGB = assemble_img(labelInnerFibre_patch, labelFullFibre_patch, labelFibreEdge_patch)
    # label_patch_RGB = conv_rgb_img(label_patch_RGB)
    image_shape = origData_patch_RGB.shape
    print(f'The number of images is {image_shape[0]} \n'
          f'The size of images is {image_shape[1:3]}')
    if show_img:
        print('Original grayscale images') 
        print(np.max(origData_patch_RGB))
        plot_imgs(1, 3, origData_patch_RGB[:,:,:,:]/255, fig_path=fig_path+'_grayscale.png', save_fig=save_fig)
        print('Masks for fibres without edges')
        plot_imgs(1, 3, label_patch_RGB[:,:,:,0], fig_path=fig_path+'_whithout_edges.png', save_fig=save_fig)
        print('Masks for whole fibres')
        plot_imgs(1, 3, label_patch_RGB[:,:,:,1], fig_path=fig_path+'_whole_fibres.png', save_fig=save_fig)
        print('Masks for fibre edges')
        plot_imgs(1, 3, label_patch_RGB[:,:,:,2], fig_path=fig_path+'_fibre_edge.png', save_fig=save_fig)
    return (origData_patch_RGB, label_patch_RGB)