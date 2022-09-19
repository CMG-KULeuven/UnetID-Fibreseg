#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Author: Rui Guo (KU Leuven), rui.guo1@kuleuven.be
Date: July 2022
"""

import numpy as np
import os
import glob
import imageio
from PIL import Image
import h5py
import pylib as py
import cv2 as cv

def import_data(inpath, file_name=False, outpath=False):
    output = []
    if file_name:
        dataset_file = os.path.join(inpath + file_name)
        data = np.array([imageio.imread(dataset_file)])
    else:
        data_files = glob.glob(os.path.join(inpath, '*.png'))
        data = np.array([imageio.imread(fname) for fname in binadata_files])
    output.append(data)
    
    if outpath:
        opdir_csv = os.path.join(outpath, 'excel')
        opdir_fig = os.path.join(outpath, 'figure')

        if not os.path.exists(opdir_csv):
            os.makedirs(opdir_csv)
        if not os.path.exists(opdir_fig):
            os.makedirs(opdir_fig)
        output.append([opdir_csv, opdir_fig])
    return output

def read_3D4Ddata(dataset_folder):
    orig_3D4DData = {}
    folder_list = os.listdir(dataset_folder)
    folder_num = len(folder_list)
    for i in range(folder_num):
        data_files = glob.glob(os.path.join(dataset_folder+folder_list[i], '*'))
        data_files = [fname for fname in data_files if fname.split('.')[-1].lower() in ['jpg', 'jpeg', 'png', 'bmp', 'tif', 'tiff']]
        orig_3D4DData[folder_list[i]] = np.array([imageio.imread(fname) for fname in data_files])
    if folder_num == 1:
        print(f'This folder saves a 3D data')
    else:
        print(f'This folder saves a 4D data and in the 4th dimension, there are {folder_num} sets of data')
        ordinal = lambda n: "%d%s" % (n,"tsnrhtdd"[(n//10%10!=1)*(n%10<4)*n%10::4])
        for i in range(folder_num):
            print(f'The {ordinal(i+1)} set of data is: {folder_list[i]}')
    return orig_3D4DData, folder_list

def save_3Ddata(path_list, data3D, save_format=['png']):
    saved_slices_num = 10
    path = py.join(path_list[0], path_list[1])
    py.mkdir(path)

    # print(np.max(data3D))
    # Save the 3D CT data into 2D slice image
    if isinstance(data3D, (list, tuple)):
        data3D = np.concatenate(data3D)
    depth = data3D.shape[0]
    if depth < saved_slices_num:
        saved_slices_num = depth
        # print(f'{" "*3} Only {saved_slices_num} slices are printed out!')

    if 'png' in save_format:
        whole_dir = py.join(path, 'whole_fibre&edge')
        py.mkdir(whole_dir)
        inner_fibre_dir = py.join(path, 'inner_fibre')
        py.mkdir(inner_fibre_dir)
        edge_dir = py.join(path, 'edge')
        py.mkdir(edge_dir)

        for i in range(saved_slices_num):
            im = Image.fromarray(data3D[i,...,1])
            im.save(whole_dir + '/slice_whole'+str(i).zfill(5)+'.png')
            im = Image.fromarray(data3D[i,...,0])
            im.save(inner_fibre_dir + '/slice_inner'+str(i).zfill(5)+'.png')
            im = Image.fromarray(data3D[i,...,2])
            im.save(edge_dir + '/slice_edge'+str(i).zfill(5)+'.png')
    
    if 'H5' in save_format:
        with h5py.File((path + '/Segmented_process.h5'), 'w') as hdf:
            hdf.create_dataset('data', data=data3D)

def conv_3Channel(img):
    dimSize = img.ndim
    if dimSize == 4:
        batch, height, width, depth = np.array(img).shape
        if depth < 3:
            dst_img = np.zeros((batch, height, width, 3))
            dst_img[:,:,:,0] = img[:,:,:,0]
            dst_img[:,:,:,1] = img[:,:,:,0]
            dst_img[:,:,:,2] = img[:,:,:,0]
        else:
            dst_img = img
    elif dimSize == 3:
        batch, height, width = np.array(img).shape
        dst_img = np.zeros((batch, height, width, 3))
        dst_img[:,:,:,0] = img[:,:,:]
        dst_img[:,:,:,1] = img[:,:,:]
        dst_img[:,:,:,2] = img[:,:,:]
    elif dimSize == 2:
        height, width = np.array(img).shape
        dst_img = np.zeros((height, width, 3))
        dst_img[:,:,0] = img
        dst_img[:,:,1] = img
        dst_img[:,:,2] = img
    return dst_img

def read_h5df_names(folder, file_id=None):
    fileName_list = os.listdir(folder)
    files_num = len(fileName_list)
    if file_id == None:
        h5df_files = []
        for i in range(files_num):
            new_fileName_list = [fname[0:-3] for fname in fileName_list]
            files = glob.glob(os.path.join(folder+fileName_list[i]))
            h5df_files.append(files[0])
    else:
        h5df_files = glob.glob(os.path.join(folder+fileName_list[file_id]))
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
    return h5df_files, new_fileName_list

def load_fromH5PY(folder, file_id=None, dtype=np.float16, file_name=None):
    # file_name: If each file (same file name) is saved in the folder, then input the file name
    # For example: Case 1: we don;t need to input the file name:
    #              data1.h5
    #              data2.h
    #              Case 2: we need to input the file name:
    #              data.h5 is in folder /data1;
    #              data.h5 is in folder /data2;
    #              For this cse, file name is data.h5(all folder have the same file name)
    h5df_files, new_fileName_list = read_h5df_names(folder, file_id=file_id)
    files_num = len(h5df_files)
    # print(h5df_files)
    orig_3D4DData = {}
    count = 0
    for h5df_file in h5df_files:
        if file_name == None:
            with h5py.File(h5df_file, 'r') as hdf:
                data = hdf.get('data')
                data3D_arr = np.array(data, dtype=dtype)
            orig_3D4DData[new_fileName_list[count]] = data3D_arr
        else: 
            with h5py.File(py.join(h5df_file, file_name), 'r') as hdf:
                data = hdf.get('data')
                data3D_arr = np.array(data, dtype=dtype)
            orig_3D4DData[new_fileName_list[count]] = data3D_arr
        count += 1
    return orig_3D4DData, new_fileName_list

def load_from_multiple_slices(dataset_folder, dataset_name_list):
    """
    Load the multiple slices from the dataset folder.
    Args:
        dataset_folder (str): Folder where the multiple slices are saved
        dataset_name_list (list): the name of slices which are used for training.
    """
    slice_num = len(dataset_name_list)
    # print(f'{slice_num} slices(images) are used for training')
    origData_list = []
    for i in range(slice_num):
        dataset_file = py.join(dataset_folder, dataset_name_list[i])
        origData = np.array(imageio.imread(dataset_file))
        if np.ndim(origData) > 2:
            origData = origData[:,:,0]
        origData_list.append(origData)
    return origData_list

def matlab_kernel(kernel):
    if kernel == 7:
        open_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(kernel,kernel))
        open_kernel[0,2] = 1
        open_kernel[0,4] = 1
        open_kernel[6,2] = 1
        open_kernel[6,4] = 1
    else:
        open_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(kernel,kernel))
    return open_kernel

def morphology(data, method='open', kernel='cross', iteration=1):
    morp_list = []
    if kernel == 'cross':
        kernel = np.array([[0,1,0],[1,1,1],[0,1,0]],np.uint8)
    elif kernel == 'full':
        kernel = np.array([[1,1,1],[1,1,1],[1,1,1]],np.uint8)
    elif kernel == 'matlab_kernel(7)':
        kernel = matlab_kernel(kernel=7)
    else:
        print("There are only two options for kernel: 'corss' or 'full'")
    if method == 'close':
        morph_method = cv.MORPH_CLOSE
    elif method == 'open':
        morph_method = cv.MORPH_OPEN
    # Save the 3D CT data into 2D slice image
    if isinstance(data, (list, tuple)):
        data = np.concatenate(data)
    ndim = np.ndim(data)
    if ndim == 3 or ndim == 2:
        res = cv.morphologyEx(data, morph_method, kernel, iterations=iteration)
        return res
    elif ndim == 4:
        depth = data.shape[0]
        res_list = [cv.morphologyEx(data[i], morph_method, kernel, iterations=iteration) for i in range(depth)]
        return np.array(res_list)

    
    