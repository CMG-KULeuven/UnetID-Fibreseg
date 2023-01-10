#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Author: Rui Guo (KU Leuven), rui.guo1@kuleuven.be
Date: July 2022
"""

import numpy as np
import os

join = os.path.join

def mkdir(paths):
    if not isinstance(paths, (list, tuple)):
        paths = [paths]
    for path in paths:
        if not os.path.exists(path):
            os.makedirs(path)
            
def normalize_8_bit(image, min_value=None, max_value=None):
    """Normalize the values in an array to the range [0, 255] and return as uint8.

    Args:
        image (numpy array): The input array.
        min_value (float, optional): The minimum value. Defaults to None 
        then the minimum value in input array wll be used. 
        max_value (float, optional): The minimum value. Defaults to None 
        then the maximum value in input array wll be used.
    
    Returns:
        Iimage (numpy array): numpy array of dtype uint8
    """    
    if min_value is None or max_value is None:
        min_value = image.min()
        max_value = image.max()
    max_value -= min_value
    image = ((image - min_value)/max_value) * 255
    image = np.clip(image, 0, 255)
    return image.astype(np.uint8)

def load_3d_dataset(dataset_folder):
    """Read 3D data from a folder.

    Args:
        dataset_folder (str): The path to the folder containing the 3D data.

    Returns:
        (array or dict): The 3D data is a numpy array or a dictionary of numpy arrays. 
        (list): The list of set names is a list of strings or an empty list.
    """    
    orig_3DData = {}
    dataset_folders = os.listdir(dataset_folder)
    num_datasets = len(dataset_folders)
    datasets_are_images = False
    image_formats = ('jpg', 'jpeg', 'png', 'bmp', 'tif', 'tiff')
    if dataset_folders[0].split('.')[-1].lower() in image_formats:
        datasets_are_images = True
        data_files = [fname for fname in glob.glob(py.join(dataset_folder+dataset_folders[i], '*'))
                      if fname.split('.')[-1].lower() in image_formats]
        orig_3DData = np.array([imageio.imread(fname) for fname in data_files])
        
    if not datasets_are_images:
        for i, folder in enumerate(dataset_folders):
            data_files = [fname for fname in glob.glob(os.path.join(dataset_folder+folder, '*'))
                            if fname.split('.')[-1].lower() in ('jpg', 'jpeg', 'png', 'bmp', 'tif', 'tiff')]
            orig_3DData[folder] = np.array([imageio.imread(fname) for fname in data_files])
            
    if num_datasets == 1 or datasets_are_images:
        print(f'This folder saves a 3D data')
    else:
        print(f'This folder saves a multiple 3D data and in the 4th dimension, there are {num_datasets} sets of data')
        ordinal = lambda n: "%d%s" % (n,"tsnrhtdd"[(n//10%10!=1)*(n%10<4)*n%10::4])
        for i, folder in enumerate(dataset_folders):
            print(f'The {ordinal(i+1)} set of data is: {folder}')
    return orig_3DData, dataset_folders

def save_3d_dataset(path_list, data3D, save_format=('png'), save_image_num=20, save_info='deflaut'):
    """Save 3D data to a folder as images or a HDF5 file.

    Args:
        path_list (list): A list containing the parent directory and the subdirectory to save the data.
        data3D (numpy array): The 3D data to save.
        save_format (list, optional): A tuple of strings specifying the formats to save the data. Default is ('png',).
        save_image_num (int, optional):  An integer specifying the number of slices to save. Defaults to 20.
        save_info (str, optional):A string specifying the type of slices to save. Can be 'default', 'inner'.
    """    
    path = py.join(path_list[0], path_list[1])
    py.mkdir(path)

    # Save the 3D CT data into 2D slice image
    if isinstance(data3D, (list, tuple)):
        data3D = np.concatenate(data3D)
    depth = data3D.shape[0]
    if save_image_num == 'Full':
        save_image_num = depth
    else:
        if depth < save_image_num:
            save_image_num = depth
            print(f'{" "*3} Only {save_image_num} slices are printed out!')

    if 'png' in save_format:
        if save_info=='default':
            whole_dir = py.join(path, 'whole_fibre&edge')
            py.mkdir(whole_dir)
            inner_fibre_dir = py.join(path, 'inner_fibre')
            py.mkdir(inner_fibre_dir)
            edge_dir = py.join(path, 'edge')
            py.mkdir(edge_dir)

            for i in range(save_image_num):
                im = Image.fromarray(data3D[i,...,1])
                im.save(whole_dir + '/slice_whole'+str(i).zfill(5)+'.png')
                im = Image.fromarray(data3D[i,...,0])
                im.save(inner_fibre_dir + '/slice_inner'+str(i).zfill(5)+'.png')
                im = Image.fromarray(data3D[i,...,2])
                im.save(edge_dir + '/slice_edge'+str(i).zfill(5)+'.png')
        
        elif save_info=='inner':
            inner_fibre_dir = py.join(path, 'inner_fibre')
            py.mkdir(inner_fibre_dir)

            for i in range(save_image_num):
                im = Image.fromarray(data3D[i,...,0])
                im.save(inner_fibre_dir + '/slice_inner'+str(i).zfill(5)+'.png')
                
        else: 
            print('We do not provide such api, please change it as default, then you can get all information')
    
    if 'H5' in save_format:
        if save_info=='default':
            with h5py.File((path + '/Segmented_process.h5'), 'w') as hdf:
                hdf.create_dataset('data', data=data3D)
        elif save_info=='inner':
            with h5py.File((path + '/Segmented_process.h5'), 'w') as hdf:
                hdf.create_dataset('data', data=data3D[...,0])

def convert_3_channel_image(img):
    """Convert an image to 3 channels.

    Args:
        img (numpy array): The input image.

    Returns:
        dst_img (numpy array): The output image with 3 channels.
    """
    img = np.array(img)
    dim_size  = img.ndim
    if dim_size == 4:
        batch, height, width, depth = img.shape
        if depth < 3:
            dst_img = np.zeros((batch, height, width, 3))
            dst_img[:,:,:,:] = img[:,:,:,0:1]
        else:
            dst_img = img
    elif dim_size == 3:
        batch, height, width = img.shape
        dst_img = np.zeros((batch, height, width, 3))
        dst_img[:,:,:,0] = img[:,:,:]
        dst_img[:,:,:,1] = img[:,:,:]
        dst_img[:,:,:,2] = img[:,:,:]
    elif dim_size == 2:
        height, width = img.shape
        dst_img = np.zeros((height, width, 3))
        dst_img[:,:,0] = img
        dst_img[:,:,1] = img
        dst_img[:,:,2] = img
    return dst_img.astype(np.uint8)

def load_hdf5_dataset_names(folder, file_id=None):
    """Read the names of H5PY files in a folder.

    Args:
        folder (str): The path to the folder containing the H5PY files.
        file_id (int, optional): The index of the H5PY file to return the name of. If not specified, the names of all
            H5PY files in the folder will be returned.

    Returns:
        hdf5_files (list): A list of H5PY file names in the specified folder.
        new_fileName_list (int): The number of H5PY files in the specified folder.
    """
    fileName_list = os.listdir(folder)
    files_num = len(fileName_list)
    if file_id == None:
        hdf5_files = []
        for i in range(files_num):
            new_fileName_list = [fname[0:-3] for fname in fileName_list]
            files = glob.glob(py.join(folder+fileName_list[i]))
            hdf5_files.append(files[0])
    else:
        hdf5_files = glob.glob(py.join(folder+fileName_list[file_id]))
        new_fileName_list = [fileName_list[file_id][:-3]]
    if files_num == 1:
        print(f'This folder saves a 3D data')
    else:
        print(f'This folder saves a 4D data and in the 4th dimension, there are {files_num} sets of data')
        ordinal = lambda n: "%d%s" % (n,"tsnrhtdd"[(n//10%10!=1)*(n%10<4)*n%10::4])
        for i in range(files_num):
            print(f'The {ordinal(i+1)} set of data is: {fileName_list[i]}')
    if file_id is not None:
        print(f'We only choose the data for the {ordinal(file_id+1)} set of data')
    
    return hdf5_files, new_fileName_list

def load_hdf5_data(folder, dtype=np.float16, file_name=None):
    """Load data from H5PY files.

    Args:
        folder (str): The path to the folder containing the H5PY files.
        dtype (numpy dtype, optional): The data type to use when loading the data from the H5PY files. Default is
            `np.float16`.
        file_name (str, optional): The name of the H5PY files to load data from. If not specified, the function will
            load data from all H5PY files in the folder.

    Returns:
        orig_3DData (dict): A dictionary containing the data from the H5PY files, where the keys are the file names and the values are
            the data.
        new_fileName_list (list): A list of the file names used as keys in the returned dictionary.
    """
    
    hdf5_files, new_fileName_list = load_hdf5_dataset_names(folder)
    orig_3DData = {}
    count = 0
    for hdf5_file in hdf5_files:
        if file_name == None:
            with h5py.File(hdf5_file, 'r') as hdf:
                data = hdf.get('data')
                data3D_arr = np.array(data, dtype=dtype)
            orig_3DData[new_fileName_list[count]] = data3D_arr
        else: 
            with h5py.File(py.join(hdf5_file, file_name), 'r') as hdf:
                data = hdf.get('data')
                data3D_arr = np.array(data, dtype=dtype)
            orig_3DData[new_fileName_list[count]] = data3D_arr
        count += 1
    return orig_3DData, new_fileName_list

# Remove the fibre at the image border:
def remove_border_fibre(img, kernel=None):
    if kernel is None:
        kernel = [[0,1,0],[1,1,1],[0,1,0]]
    mask = img == 255
    labeled_mask, _ = ndimage.label(mask, structure=kernel)
    cleared_img = clear_border(labeled_mask)
    cleared_img = (cleared_img > 0.9) * 255
    return cleared_img