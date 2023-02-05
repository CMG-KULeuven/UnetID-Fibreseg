#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import cv2 as cv

def generate_disk_kernel(disk_radiu=4):
    # https://www.mathworks.com/help/images/ref/strel.html
    if disk_radiu == 4:
        morph_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(7, 7))
        morph_kernel[0,2] = 1
        morph_kernel[0,4] = 1
        morph_kernel[6,2] = 1
        morph_kernel[6,4] = 1
    elif disk_radiu == 3:
        morph_kernel = np.ones((5,5),np.uint8)
    elif disk_radiu == 2:
        morph_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(5, 5))
        morph_kernel[1,0] = 0
        morph_kernel[3,0] = 0
        morph_kernel[1,4] = 0
        morph_kernel[3,4] = 0
    elif disk_radiu == 1:
        morph_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(3, 3))
    elif disk_radiu == 5:
        morph_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(9, 9))
        morph_kernel[0,2] = 1
        morph_kernel[0,3] = 1
        morph_kernel[0,5] = 1
        morph_kernel[0,6] = 1
        morph_kernel[2,0] = 1
        morph_kernel[2,8] = 1
        morph_kernel[6,0] = 1
        morph_kernel[6,8] = 1
        morph_kernel[8,2] = 1
        morph_kernel[8,3] = 1
        morph_kernel[8,5] = 1
        morph_kernel[8,6] = 1
    else:
        print('We only provide the disk kernel when the radius is equal to 1,2,3,4,5')
    return morph_kernel

def generate_ellipse_kernel(ksize=4):
    # https://docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html#gac342a1bb6eabf6f55c803b09268e36dc
    morph_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(ksize, ksize))
    return morph_kernel

def apply_morphological_operation(data, method='open', kernel={"kernel_shape":"disk", "kernel_radius":4}, iteration=1):
    """ Applies morphological operations to a 2D or 3D array.

    Args:
        data (numpy.ndarray): The input array.
        method (str, optional): The morphological operation to perform as one of 'open' or 'close'. Defaults to 'open'.
        kernel (dict, optional): The morphological structuring element. Defaults to {"kernel_shape":"disk", "kernel_radius":4}.
        iteration (int, optional): The number of times to apply the morphological operation. Default is 1

    Returns:
        (array): The images after applying the morphological operation.
        
    Note for kernel:
    For post process, it is important to choose an appropriate parameter for kernel size.
    The default "kernel" is: {"kernel_shape":"disk", "kernel_radius":4}
    you can set the kernel radius as 1, 2, 3, 4, 5 according your requirements.
    Of course, you can also choose the ellipse kernel shape, now you should input the kernel size instand of kernel radius, 
    like this "kernel": {"kernel_shape":"ellipse", "kernel_size":4}. The size you can choose any value as your requirements.
    """    
    morp_list = []
    if kernel == 'cross':
        morph_kernel = np.array([[0,1,0],[1,1,1],[0,1,0]],np.uint8)
    elif kernel == 'full':
        morph_kernel = np.array([[1,1,1],[1,1,1],[1,1,1]],np.uint8)
    elif kernel["kernel_shape"] == "disk":
        try:
            morph_kernel = generate_disk_kernel(disk_radiu= kernel["kernel_radius"])
        except:
            print('Please check the kernel information, ensuring you are using  kernel_radius not kernel_size')
        
    elif kernel["kernel_shape"] == "ellipse":
        try:
            morph_kernel = generate_ellipse_kernel(kernel_size= kernel["kernel_size"])
        except:
            print('Please check the kernel information, ensuring you are using kernel_size not kernel_radius')
    if method == 'close':
        morph_method = cv.MORPH_CLOSE
    elif method == 'open':
        morph_method = cv.MORPH_OPEN
    # Save the 3D CT data into 2D slice image
    if isinstance(data, (list, tuple)):
        data = np.concatenate(data)
    ndim = np.ndim(data)
    if ndim == 3 or ndim == 2:
        res = cv.morphologyEx(data, morph_method, morph_kernel, iterations=iteration)
        return res
    elif ndim == 4:
        depth = data.shape[0]
        res_list = [cv.morphologyEx(data[i], morph_method, morph_kernel, iterations=iteration) for i in range(depth)]
        return np.array(res_list)

def slide_window(segm_img, step, half_step, small_patch_pred, i, j):
    """
    Apply a small patch prediction to an image using a sliding window.
    
    Args:
        segm_img (torch.Tensor): The image to be updated.
        step (int): The size of the patch.
        half_step (int): Half of the patch size.
        small_patch_pred (torch.Tensor): The prediction for a small patch.
        i (int): The row index of the top left corner of the patch in the image.
        j (int): The column index of the top left corner of the patch in the image.
        
    Returns:
        segm_img (torch.Tensor): The updated image.
    """
    if i == 0 and j ==0:
        segm_img[:, :, i:i+step+half_step, j:j+step+half_step] \
            += small_patch_pred[:, :, 0:step+half_step, 0:step+half_step]
    elif i == 0 and j != 0:
        segm_img[:, :, i:i+step+half_step, j+half_step:j+step+half_step] \
            += small_patch_pred[:, :, 0:step+half_step, half_step:step+half_step]
    elif i != 0 and j == 0:
        segm_img[:, :, i+half_step:i+step+half_step, j:j+step+half_step] \
            += small_patch_pred[:, :, half_step:step+half_step, 0:step+half_step]
    else:
        segm_img[:, :, i+half_step:i+step+half_step, j+half_step:j+step+half_step] \
            += small_patch_pred[:, :, half_step:step+half_step, half_step:step+half_step]
    
    return segm_img
    