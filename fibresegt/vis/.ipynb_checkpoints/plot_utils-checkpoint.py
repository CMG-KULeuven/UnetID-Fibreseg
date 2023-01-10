#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Author: Rui Guo (KU Leuven), rui.guo1@kuleuven.be
Date: July 2022
"""

import matplotlib.pyplot as plt 
import numpy as np
import seaborn as sns
from skimage.color import label2rgb
from skimage.measure import label
import cv2 as cv


def plot_multiple_images(row, col, images, fig_path=None, save_fig=False):
    """Plot multiple images in a single figure.

    Args:
        row (int): number of rows in the subplot.
        col (int):  number of collumns in the subplot.
        images (list): list of images to plot.
        fig_path (bool, optional): whether to save the figure. Defaults to False.
        save_fig (bool, optional): path to save the figure. Defaults to None.
    """    
    for i in range(row*col):
        plt.subplot(row, col, 1+i)
        # Turn off the aixs
        if save_fig == True:
            plt.axis('off')
        plt.imshow(images[i], cmap='gray')
    
    plt.tight_layout()
    if save_fig == True:
        plt.savefig(fig_path, dpi=300)
    plt.show()

def visualize_3d_data(data_list, set_id, visulize_plane_range, overlay=False, show_model=None):
    """Visualize the 3D data.

    Args:
        data_list (_list_):  A list to save the 3D data and segmented results.
        set_id (_str or int_): specify which set of data you want to see
        visulize_plane_range (_list_): Specify which plane and plane size of the data you want to see
        overlay (_bool_): Set to True if you want to see the overlay results
    """ 
    if overlay:
        data_num = len(data_list)
        if data_num == 1:
            print(f'Please check whether you input the segmented results in the data')
    
    slice_range, height_range, width_range = visulize_plane_range
    slice_num = len(slice_range)
    height_num = len(height_range)
    width_num = len(width_range)
    if slice_num == 1:
        for data in data_list:
            data = data[set_id]
            data2d = []
            if isinstance (data, (tuple, list)):
                data = data[slice_range[0]]
            data2d.append(data[slice_range[0],:,:])
        visulize_area = [height_range, width_range]
        if overlay:
            overlay(data2d[0], data2d[1], visulize_area, show_model=show_model)
        else:
            plt.imshow(data2d[0][height_range[0]:height_range[1], width_range[0]:width_range[1]], cmap='gray')
            plt.show()
    if height_num == 1:
        for data in data_list:
            data = data[set_id]
            data2d = []
            if isinstance(data, (tuple, list)):
                data = np.concatenate(data)
                print(data.shape)
            data2d.append(data[:,height_range[0],:])
        visulize_area = [slice_range, width_range]
        if overlay:
            overlay(data2d[0], data2d[1], visulize_area, show_model=show_model)
        else:
            plt.imshow(data2d[0][slice_range[0]:slice_range[1], width_range[0]:width_range[1]], cmap='gray')
            plt.show()
    if width_num == 1:
        for data in data_list:
            data = data[set_id]
            data2d = []
            if isinstance(data, (tuple, list)):
                data = np.concatenate(data)
                print(data.shape)
            data2d.append(data[:,:,width_range[0]])
        visulize_area = [slice_range, height_range]
        if overlay:
            overlay(data2d[0], data2d[1], visulize_area, show_model=show_model)
        else:
            plt.imshow(data2d[0][slice_range[0]:slice_range[1], height_range[0]:height_range[1]], cmap='gray')
            plt.show()
    return

def overlay_images(grayimg, binaryimg, remove_border_fibres=True, save_fig=True, save_path=None, enhance=True, opacity=0.5):
    """Overlay a binary image on top of a grayscale image.

    Args:
        grayimg (array): Grayscale image.
        binaryimg (array): Binary image.
        remove_border_fibres (bool, optional): Whether to remove border fibers. Defaults to True.
        save_fig (bool, optional): Whether to save the resulting image. Defaults to True.
        save_path (str, optional): Path to save resulting image. Defaults to None.
        enhance (bool, optional): Whether to enhance the grayscale image. Defaults to True.
        opacity (float, optional): Transparency level of binary image when overlaid on grayscale image. Defaults to 0.5.
    """    
    palette = sns.color_palette()
    if remove_border_fibres:
        binaryimg = mc.remove_border_fibre(binaryimg)
        file_name = '_remove_border_fibres'
    else:
        file_name = '_with_border_fibres'
    if enhance:
        grayimg = cv.equalizeHist(grayimg)
    label_image = label(binaryimg)
    image_label_overlay = label2rgb(label_image, colors = palette, image=grayimg, 
                                    alpha=opacity,bg_label=0, kind='overlay')
    if save_fig:
        # plt.imshow(image_label_overlay)
        fig_path = save_path[0:-4]+file_name+'.png'
        plt.imsave(fig_path, image_label_overlay,dpi=300)