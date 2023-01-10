"""
Author: Rui Guo (KU Leuven), rui.guo1@kuleuven.be
Date: Nov 2022
"""

import numpy as np
from scipy import ndimage
from skimage import measure
from skimage.segmentation import clear_border
from fibresegt.data.data_utils import write_to_txt, write_to_excel
from PIL import Image

def IoU_metric(im1, im2, empty_score=1.0, save_data=True, save_path=None, access_mode='write'):
    """Intersection-Over-Union (IoU), also known as the Jaccard Index"""
    im1 = np.asarray(im1).astype(np.bool)
    im2 = np.asarray(im2).astype(np.bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    im1 = im1 > 0.5
    im2 = im2 > 0.5

    im_sum = im1.sum() + im2.sum()
    if im_sum == 0:
        return empty_score

    # Compute IOU score
    intersection = np.logical_and(im1, im2)
    union = np.logical_or(im1, im2)
    iou_score = np.sum(intersection) / np.sum(union)
    
    print_title = f'--------------------------------------IoU score--------------------------------------/n'
    if save_data:
        print(print_title)
        print_results = f'The IoU score is: {iou_score}'
        print(print_results)
        write_to_txt(print_title, save_path=save_path, access_mode=access_mode)
        write_to_txt(print_results, save_path=save_path, access_mode='append')
    return iou_score

# Remove the fibre at the image border:
def remove_border_fibre(img, kernel=None):
    if kernel is None:
        kernel = [[0,1,0],[1,1,1],[0,1,0]]
    mask = img == 255
    labeled_mask, _ = ndimage.label(mask, structure=kernel)
    cleared_img = clear_border(labeled_mask)
    cleared_img = (cleared_img > 0.9) * 255
    return cleared_img

# Find the properties for the labeled mask
def extrac_prop(img):
    struc = [[0,1,0],[1,1,1],[0,1,0]]
    mask = img == 255
    labeled_mask, num_labels = ndimage.label(mask, structure=struc)
    # Find the properties for the labeled mask
    regions_prop = measure.regionprops(labeled_mask, img)
    return regions_prop

# Extract the fibre center
def extract_fibre_center(img):
    fibre_center_list = []
    regions_prop = extrac_prop(img)
    fibre_num = len(regions_prop)
    for i in range(fibre_num):
        y0, x0 = regions_prop[i].centroid
        fibre_center_list.append([x0, y0])
    return fibre_center_list

# Calculate the fibre equavilent diameter 
def equiv_diam_func(img, unit, remove_border_fibres=True, kernel=None, 
                    save_data=True, save_path=None, access_mode='append'):
    if kernel is None:
        kernel = [[0,1,0],[1,1,1],[0,1,0]]
    if remove_border_fibres:
        img = remove_border_fibre(img, kernel=kernel)
        print_var = 'After removing the fibres at the image border, '
    else:
        print_var = 'With incompleted fibres at the image border, '
    regions_prop = extrac_prop(img)
    fibre_num = len(regions_prop)
    equiv_diam_mm_list = []
    for i in range(fibre_num):
        equiv_diam = regions_prop[i].equivalent_diameter
        equiv_diam_mm_list.append(equiv_diam*unit)
    print_title = f'--------------------------------------fibre equivalent diameter--------------------------------------/n'
    if remove_border_fibres:
        print_results = f'After removing the fibres at the image border, the number of fibers is: {fibre_num}'
        if save_data:
            print(print_title)
            print(print_results)
            write_to_txt(print_results, save_path=save_path, access_mode=access_mode)
    print_results = f'{print_var} the mean and standard deviation of fibre equivalent diameter is: \
                    {np.mean(equiv_diam_mm_list)}, {np.std(equiv_diam_mm_list)}'
    if save_data:
        print(print_title)
        print(print_results)
        write_to_txt(print_title, save_path=save_path, access_mode=access_mode)
        write_to_txt(print_results, save_path=save_path, access_mode='append')
        write_to_txt(str(equiv_diam_mm_list), save_path=save_path, access_mode='append')
        write_to_excel(data=equiv_diam_mm_list, path=save_path[:-4], header=['fibre_equiv_dia'])
        im = Image.fromarray(img.astype(np.uint8))
        im.save(save_path[0:-4]+'_removed_border_fibres.tif')
        im = Image.fromarray(img.astype(np.uint8))
        im.save(save_path[0:-4]+'.tif')
    return equiv_diam_mm_list, fibre_num

# Calculate the fibre numbers including the fibres at the image border:
def count_fibres_func(img, kernel=None, save_data=True, save_path=None, access_mode='write'):
    regions_prop = extrac_prop(img)
    fiber_num = len(regions_prop)
    print_title = f'--------------------------------------fibre number--------------------------------------/n'
    
    print_results = f'The total number of fibers in this image is: {fiber_num}'
    if save_data:
        print(print_title)
        print(print_results)
        write_to_txt(print_title, save_path=save_path, access_mode=access_mode)
        write_to_txt(print_results, save_path=save_path, access_mode='append')
    return fiber_num

# Calculate the fibre volume fraction
def volume_fraction(img, thred=0.5, save_data=True, save_path=None, access_mode='append'):
    height, width = img.shape[0:2]
    obj_area = np.sum(img>=thred)
    obj_fv = obj_area/(height*width)
    print_title = f'------------------------------------fibre volume fraction-----------------------------------/n'
    print_results = f'The fibre volume fraction for this image is: {obj_fv}'
    if save_data:
        print(print_title)
        print(print_results)
        write_to_txt(print_title, save_path=save_path, access_mode=access_mode)
        write_to_txt(print_results, save_path=save_path, access_mode='append')
    return obj_fv
    
# Calculate the fibre flatness
def fibre_aspect_ratio_func(img, remove_border_fibres=True, kernel=None, save_data=True, 
                            save_path=None, access_mode='append', epsilon=1e-9):
    if remove_border_fibres:
        img = remove_border_fibre(img, kernel=kernel)
        print_var = 'After removing the fibres at the image border, '
    else:
        print_var = 'With incompleted fibres at the image border, '
    regions_prop = extrac_prop(img)
    fibre_num = len(regions_prop)
    aspect_ratio_list = []
    for i in range(fibre_num):
        props = regions_prop[i]
        minor_axis_length_ = props['minor_axis_length']
        major_axis_length_ = props['major_axis_length']
        if minor_axis_length_ ==0:
            minor_axis_length_ = 1
        if major_axis_length_ ==0:
            major_axis_length_ = 1
        aspect_ratio_list.append(major_axis_length_/minor_axis_length_)
    print_title = f'--------------------------------------fibre aspect ratio--------------------------------------/n'
    print_results = f'{print_var} the mean and standard deviation of fibre flatness is: {np.mean(aspect_ratio_list)}, \
                        {np.std(aspect_ratio_list)}'
    if save_data:
        print(print_title)
        print(print_results)
        write_to_txt(print_title, save_path=save_path, access_mode=access_mode)
        write_to_txt(print_results, save_path=save_path, access_mode='append')
        write_to_txt(str(aspect_ratio_list), save_path=save_path, access_mode='append')
        write_to_excel(data=aspect_ratio_list, path=save_path[:-4], header=['fibre_flatness'])
    return aspect_ratio_list

def metrics_eval(img, label=None, img2=None, remove_border_fibres=True, save_data=False, save_path=None, evaluate_methods=None):
    # Calculate the metrics for a single image
    fibres_prop = {}
    if save_data:
        write_to_txt('Evaluate', save_path=save_path, access_mode='write')
    for eva_method in evaluate_methods:
        if eva_method == 'IoU_score':
            IoU_score = IoU_metric(im1=img, im2=label, empty_score=1.0,
                          save_data=save_data, save_path=save_path, access_mode='append')
            fibres_prop['IoU_score'] = IoU_score
        if eva_method == 'fibre_num':
            fibre_num = count_fibres_func(img=img, save_data=save_data, save_path=save_path, access_mode='append')
            fibres_prop['fibre_num'] = fibre_num
        if eva_method == 'fibreID':
            fibre_num = count_fibres_func(img=img, save_data=save_data, save_path=save_path, access_mode='append')
            fibres_prop['fibreID'] =  np.arange(fibre_num)
        if eva_method == 'fibre_centerCoords':
            fibre_center = extract_fibre_center(img)
            fibres_prop['fibre_centerCoords'] = fibre_center

        if eva_method == 'fibre_equiv_dia' or 'fibre_diameter':
            fibre_equiv_dia, _ = equiv_diam_func(img=img, unit=1, remove_border_fibres=remove_border_fibres,
                                                save_data=save_data, save_path=save_path, access_mode='append')
            fibres_prop['fibre_diameter'] = fibre_equiv_dia
        if eva_method == 'fibre_volume_fraction':
            fv = volume_fraction(img=img, thred=0.5, save_data=save_data, save_path=save_path, access_mode='append')
            fibres_prop['fibre_volume_fraction'] = fv
        if eva_method == 'fibre_flatness' or 'fibre_AspectRatio':
            aspect_ratio_list = fibre_aspect_ratio_func(img=img, remove_border_fibres=remove_border_fibres, save_data=save_data, save_path=save_path, access_mode='append')
            fibres_prop['fibre_AspectRatio'] = aspect_ratio_list


    if save_data:
        print('Data is saved!')
    return fibres_prop

def calcu_metrics_eval(segm_3Ddata, label=None, save_data=True, save_path=None, evaluate_methods=None):
    """_summary_

    Args:
        segm_3Ddata (array): Segmented images
        label (array, optional): Manual labelled masks. Defaults to None.
        save_data (bool, optional): A flag to indicate whether to save the results of the calculations to a file. Defaults to True.
        save_path (str, optional): The file path where the results should be saved if the save_data flag is set to True. Defaults to None.
        evaluate_methods (_type_, optional): Some methods for evaluating properties of the fibres. Defaults to None.
    """    
    slice_num = segm_3Ddata.shape[0]
    print(f'Dimenstion for this data along the thickness {slice_num}')
    fibres_prop_fibre_num = []
    fibres_prop_fibre_diameter = []
    fibres_prop_fibre_AspectRatio = []
    for i in range(slice_num):
        fibres_prop_slice = metrics_eval(img=segm_3Ddata[i], label=label, evaluate_methods=evaluate_methods)
        fibres_prop_fibre_num.append(fibres_prop_slice['fibre_num'])
        fibres_prop_fibre_diameter.append(fibres_prop_slice['fibre_diameter'])
        fibres_prop_fibre_AspectRatio.append(fibres_prop_slice['fibre_AspectRatio'])
    
    fibres_prop_fibre_diameter = [item for sublist in fibres_prop_fibre_diameter for item in sublist]
    fibres_prop_fibre_AspectRatio = [item for sublist in fibres_prop_fibre_AspectRatio for item in sublist]
    print_results = f'the mean and standard deviation of fibre_num for multiple slices is : \
                    {np.mean(fibres_prop_fibre_num)}, {np.std(fibres_prop_fibre_num)}'
    # Save the data
    if save_data:
        write_to_txt(print_results, save_path=save_path, access_mode='append')
        print_results = f'the mean and standard deviation of fibre_diameter for all complete fibres in multiple slices is : \
                        {np.mean(fibres_prop_fibre_diameter)}, {np.std(fibres_prop_fibre_diameter)}'
        write_to_txt(print_results, save_path=save_path, access_mode='append')
        print_results = f'the mean and standard deviation of fibre_AspectRatio  for all complete fibres in multiple slices is : \
                        {np.mean(fibres_prop_fibre_AspectRatio)}, {np.std(fibres_prop_fibre_AspectRatio)}'
        write_to_txt(print_results, save_path=save_path, access_mode='append')

        write_to_excel(data=fibres_prop_fibre_num, path=save_path[:-4], header=['fibre_num'])
        write_to_excel(data=fibres_prop_fibre_diameter, path=save_path[:-4], header=['fibre_equiv_dia'])
        write_to_excel(data=fibres_prop_fibre_AspectRatio, path=save_path[:-4], header=['fibre_flatness'])
