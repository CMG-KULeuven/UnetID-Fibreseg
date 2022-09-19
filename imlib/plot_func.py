#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Author: Rui Guo (KU Leuven), rui.guo1@kuleuven.be
Date: July 2022
"""

import matplotlib.pyplot as plt 
import numpy as np
import matplotlib.ticker as mtick
import imlib as im

def plot_img(row, col, data, fig_path=None, save_fig=False):
    # this plot is only suit to show 6*3 images
    fig, axs = plt.subplots(row, col, sharex=True, sharey=True, figsize=(10, 10))
    fig.set_canvas(plt.gcf().canvas) # In order 
    num = 0
    
    for i in range(row):
        for j in range(col):
            ax = axs[i, j]
            ax.imshow(data[num], cmap='gray')
            ax.axis('off')
            num += 1
    plt.tight_layout()
    if save_fig == True:
        plt.savefig(fig_path, dpi=300)
    plt.show()



def plot_multi_imgs(row, col, images, fig_path=None, save_fig=False):
    for i in range(row*col):
        plt.subplot(row, col, 1+i)
        # Turn off the aixs
        # plt.axis('off')
        plt.imshow(images[i], cmap='gray')
    
    plt.tight_layout()
    if save_fig == True:
        plt.savefig(fig_path, dpi=300)
    plt.show()

def overlay(grayscale_img, mask_img, visulize_area, show_model='inner fibre'):
    """ overlay the grayscale images and segmented results

    Returns:
        _type_: _description_
    """ 
    h_range, w_range = visulize_area
    data_rgb = im.conv_3Channel(grayscale_img[h_range[0]:h_range[1], w_range[0]:w_range[1]])/255
    
    data_size = grayscale_img
    if show_model == 'inner fibre':
        mask_img = mask_img[h_range[0]:h_range[1], w_range[0]:w_range[1], 0]
    elif show_model == 'whole fibre':
        mask_img = mask_img[h_range[0]:h_range[1], w_range[0]:w_range[1], 1]
    elif show_model == 'fibre edge':
        mask_img = mask_img[h_range[0]:h_range[1], w_range[0]:w_range[1], 2]
    else:
        print('Please specify the show model correctly!')
    data_rgb[mask_img >= 0.1] = [0,0,1]
    plt.imshow(data_rgb)
    return data_rgb

def visulize_3D4Ddata(data_list, set_id, visulize_plane_range, overlay=False, show_model=None):
    """visulize_3D4Ddata: Visualize the 3D4D data in 2D images

    Args:
        data_list (_list_): a list to save the 3D4Ddata and segmented results if have 
        set_id (_str or int_): specify which set of data you want to see
        visulize_plane_range (_list_): Specify which plane and plane size of the data you want to see
        overlay (_bool_): Set True if you want to see the overlay results
    Returns:
        None: no return data
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

def plot_inter_img(n, data, fig_path=None, save_fig=False):
    plt.figure(figsize=(20, 10)) 
    for i in range(n):
        plt.subplot(1, n, 1+i)
        # turn off the axis
        plt.axis('off')
        plt.imshow(data[i, :, :], cmap='gray')
    plt.tight_layout()
    if save_fig == True:
        plt.savefig(fig_path, dpi=300)
    plt.show()


def plot_line(xticks, yticks, data_x, data_y,  xlabel, ylabel, annotate=True, save_fig=False, fig_path=None):
    plt.plot(data_x, data_y, '-o',
            markersize=12, linewidth=4,
            markerfacecolor='steelblue',
            markeredgecolor='steelblue',
            markeredgewidth=1) # line (-), circle marker (o), black (k)
    plt.xticks(xticks)
    plt.yticks(yticks)
    plt.xlabel(xlabel, fontsize=16, labelpad=10)
    plt.ylabel(ylabel, fontsize=16, labelpad=10)
    # show the value
    if annotate == True:
        for i,j in zip(data_x,data_y):
            inplot = plt.annotate(str(round(j,2))
                                  ,xy=(i,j+1)
                                  ,fontsize=14)
    
    plt.minorticks_off()
    
    plt.rcParams.update({'font.size': 12}) # This is for the size of axis
    plt.rcParams['axes.linewidth'] = 1.5 # set the value globally
    plt.rcParams['ytick.major.pad'] = 5 # Control the distance between axis and axis value
    plt.rcParams['xtick.major.pad'] = 10
    plt.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=True,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=True) # labels along the bottom edge are off
    plt.tick_params(
        axis='y',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        left=True,      # ticks along the bottom edge are off
        right=False)         # ticks along the top edge are off
    # here we modify the axes, specifying min and max for x and y axes.
    x_margin = 3
#     plt.axis(xmin=0, xmax=xticks[-1]+xticks[0]+x_margin, ymin=yticks[0], ymax=yticks[-1])
    # fllowing for str of x-vlaue
    x_margin = 3
#     plt.axis(xmin=x_margin, xmax=x_margin, ymin=yticks[0], ymax=yticks[-1])
    plt.tight_layout()
    if save_fig == True:
        plt.savefig(fig_path, dpi=300)
    plt.show()


def plot_multiple_line(xticks, yticks, data_dic,  xlabel, ylabel, save_fig=False, fig_path=None):
    # data_dic should be a dictionary {labe:[[data_x], [data_y]]}
    # We only make a marker for only three different kinds of data.
    marker = ['o', '^', 'P']
    i = 0
    for label in data_dic:
        data_temp = data_dic[label]
        data_x = data_temp[0]
        data_y = data_temp[1]
        if i < 3:
            v # line (-), circle marker (o), black (k)
        else:
            plt.plot(data_x, data_y, '-',
                    linewidth=4,
                    label=label) # line (-), circle marker (o), black (k)
        i += 1
    plt.xticks(xticks)
    plt.yticks(yticks)
    plt.xlabel(xlabel, fontsize=16, labelpad=10)
    plt.ylabel(ylabel, fontsize=16, labelpad=10)
    
    plt.legend(loc = 'upper right')
    
    plt.minorticks_off()
    
    plt.rcParams.update({'font.size': 12}) # This is for the size of axis
    plt.rcParams['axes.linewidth'] = 1.5 # set the value globally
    plt.rcParams['ytick.major.pad'] = 5 # Control the distance between axis and axis value
    plt.rcParams['xtick.major.pad'] = 10
    plt.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=True,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=True) # labels along the bottom edge are off
    plt.tick_params(
        axis='y',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        left=True,      # ticks along the bottom edge are off
        right=False)         # ticks along the top edge are off
    # here we modify the axes, specifying min and max for x and y axes.
    x_margin = 3
#     plt.axis(xmin=0, xmax=xticks[-1]+xticks[0]+x_margin, ymin=yticks[0], ymax=yticks[-1])
    # fllowing for str of x-vlaue
    x_margin = 3
#     plt.axis(xmin=x_margin, xmax=x_margin, ymin=yticks[0], ymax=yticks[-1])
    plt.tight_layout()
    if save_fig == True:
        plt.savefig(fig_path, dpi=300)
    plt.show()


def plot_bar(data, xlabel, ylabel, annotate=False, save_fig=False, fig_path=None):
    data_x = list(data.keys())
    data_y = list(data.values())
    data = {'data_x':data_x,
            'data_y':data_y}
    df = pd.DataFrame(data)
    fig = plt.figure(figsize = (5, 5))
    # creating the bar plot
    plt.bar(data_x, data_y, color ='steelblue', width = 0.4)
#     plots = sns.barplot(x="data_x", y="data_y", data=df, color ='steelblue', height=4, aspect=.7)
        # show the value
    if annotate == True:
        for i,j in zip(data_x,data_y):
            inplot = plt.annotate(str(round(j,2))
                                  ,xy=(i,j+1)
                                  ,fontsize=14)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
    plt.minorticks_off()
    
    plt.rcParams.update({'font.size': 12}) # This is for the size of axis
    plt.rcParams['axes.linewidth'] = 1.5 # set the value globally
    plt.rcParams['ytick.major.pad'] = 5 # Control the distance between axis and axis value
    plt.rcParams['xtick.major.pad'] = 10
    plt.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=True,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=True) # labels along the bottom edge are off
    plt.tick_params(
        axis='y',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        left=True,      # ticks along the bottom edge are off
        right=False)         # ticks along the top edge are off
    plt.tight_layout()
    if save_fig == True:
        plt.savefig(fig_path, dpi=300)
        
    plt.show()


def plot_bar_error(data_x, data_y, error, xticklabels, xlabel, ylabel, save_fig=False, fig_path=None):
    fig = plt.figure(figsize = (5, 5))
    plt.bar(data_x, data_y, yerr=error, align='center', color ='steelblue', alpha=1.0, width = 0.4, capsize=10)
    
    plt.xticks(data_x, xticklabels)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
    plt.minorticks_off()
    
    plt.rcParams.update({'font.size': 12}) # This is for the size of axis
    plt.rcParams['axes.linewidth'] = 1.5 # set the value globally
    plt.rcParams['ytick.major.pad'] = 5 # Control the distance between axis and axis value
    plt.rcParams['xtick.major.pad'] = 10
    plt.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=True,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=True) # labels along the bottom edge are off
    plt.tick_params(
        axis='y',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        left=True,      # ticks along the bottom edge are off
        right=False)         # ticks along the top edge are off
    plt.tight_layout()
    if save_fig == True:
        plt.savefig(fig_path, dpi=300)
    
    plt.show()


def plot_mix_bar_error(data_x_error, data_y_error, error, data_x, data_y, xlabel, ylabel, grid=True, save_fig=False, fig_path=None):
    fig = plt.figure(figsize = (5, 5))
    len_error = len(data_x_error)
    x_pos = np.arange(0, len_error, 1)
    plt.bar(data_x_error, data_y_error, yerr=error, align='center', color ='steelblue', alpha=1.0, width = 0.4, capsize=10)
    len_data = len(data_x)
    x_pos = np.arange(len_error, len_error+len_data, 1)
    plt.bar(data_x, data_y, color ='steelblue', alpha=1.0, width = 0.4)
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
    plt.minorticks_off()
    
    plt.rcParams.update({'font.size': 12}) # This is for the size of axis
    plt.rcParams['axes.linewidth'] = 1.5 # set the value globally
    plt.rcParams['ytick.major.pad'] = 5 # Control the distance between axis and axis value
    plt.rcParams['xtick.major.pad'] = 10
    plt.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=True,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=True) # labels along the bottom edge are off
    plt.tick_params(
        axis='y',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        left=True,      # ticks along the bottom edge are off
        right=False)         # ticks along the top edge are off
    
    if grid:
        plt.grid(True)
    plt.tight_layout()
    if save_fig == True:
        plt.savefig(fig_path, dpi=300)
    
    plt.show()


def plot_problty(data, bins_size, xticks, yticks, xlabel, ylabel, title, margin, save_fig=False, fig_path=None, show_x_percentage=False):
    data_num = len(data)
    n, bins = np.histogram(data, bins=bins_size, range=None, normed=None, weights=None, density=None)
    mid_bins = []
    for i in range(len(bins)-1):
        mid_bins.append((bins[i+1] + bins[i])/2.0)
        width = bins[i+1] - bins[i]
    width = width/1 # As we solved is a probability at one point, the width should be small
    problty = n/data_num
    print(data_num)
    color = (0.2980392156862745, 0.4470588235294118, 0.6901960784313725)
    plt.bar(mid_bins, problty, color=color, edgecolor='black', alpha=1.0, width =width)
    plt.xticks(xticks) # Set the
    plt.yticks(yticks)
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1, decimals=0)) # Show percentage
    if show_x_percentage:
        plt.gca().xaxis.set_major_formatter(mtick.PercentFormatter(1, decimals=0)) # Show percentage
    plt.xlabel(xlabel, fontsize=16, labelpad=10)
    plt.ylabel(ylabel, fontsize=18, labelpad=10)
#     plt.title(title, fontsize=14)

    plt.minorticks_off()
#     plt.figure(figsize=(10,10))
    plt.rcParams.update({'font.size': 12}) # This is for the size of axis
    plt.rcParams['axes.linewidth'] = 1.0 # set the value globally
    plt.rcParams['ytick.major.pad'] = 5 # Control the distance between axis and axis value
    plt.rcParams['xtick.major.pad'] = 10
    plt.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=True,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=True) # labels along the bottom edge are off
    plt.tick_params(
        axis='y',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        left=True,      # ticks along the bottom edge are off
        right=False)         # ticks along the top edge are off
    plt.tight_layout()
    
    x_min_margin, x_max_margin, y_min_margin, y_max_margin = margin
    plt.axis(xmin=xticks[0] + x_min_margin, 
             xmax=xticks[-1] + x_max_margin, 
             ymin=yticks[0] + y_min_margin, 
             ymax=yticks[-1] + y_max_margin)
    plt.tight_layout()
    if save_fig == True:
        plt.savefig(fig_path, dpi=300)
    plt.show()
    return mid_bins, problty



def plot_line_mark(xticks, yticks, data_x, data_y,  xlabel, ylabel, annotate=True, save_fig=False, fig_path=None):
    color = (0.2980392156862745, 0.4470588235294118, 0.6901960784313725),

    if annotate:
        plt.plot(data_x, data_y, '-o',
                markersize=12, linewidth=4,
                markerfacecolor=color,
                markeredgecolor=color,
                markeredgewidth=1) # line (-), circle marker (o), black (k)

        # show the value
        for i,j in zip(data_x,data_y):
            inplot = plt.annotate(str(round(j,2))
                                  ,xy=(i,j+1)
                                  ,fontsize=14)
    else:
        plt.plot(data_x, data_y,
                 linewidth=2) # line (-), circle marker (o), black (k)
        
    plt.plot((xticks[0], xticks[-1]), (1, 1), 'k-')
    plt.xticks(xticks)
    plt.yticks(yticks)
    plt.xlabel(xlabel, fontsize=16, labelpad=10)
    plt.ylabel(ylabel, fontsize=16, labelpad=10)
    
    plt.minorticks_off()
    
    plt.rcParams.update({'font.size': 12}) # This is for the size of axis
    plt.rcParams['axes.linewidth'] = 1.5 # set the value globally
    plt.rcParams['ytick.major.pad'] = 5 # Control the distance between axis and axis value
    plt.rcParams['xtick.major.pad'] = 10
    plt.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=True,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=True) # labels along the bottom edge are off
    plt.tick_params(
        axis='y',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        left=True,      # ticks along the bottom edge are off
        right=False)         # ticks along the top edge are off
    # here we modify the axes, specifying min and max for x and y axes.
    x_margin = 3
#     plt.axis(xmin=0, xmax=xticks[-1]+xticks[0]+x_margin, ymin=yticks[0], ymax=yticks[-1])
    # fllowing for str of x-vlaue
    x_margin = 3
#     plt.axis(xmin=x_margin, xmax=x_margin, ymin=yticks[0], ymax=yticks[-1])
    plt.tight_layout()
    if save_fig == True:
        plt.savefig(fig_path, dpi=300)
    plt.show()


def plot_line_error(xticks, yticks, data_x, data_y, lower_err, upper_err, sd_array, xlabel, ylabel, margin, mark=True, save_fig=False, fig_path=None, errortype='ContinueError'):
#     plt.plot(data_x, data_y, 'or')
    if errortype == 'ContinueError':
        plt.plot(data_x, data_y, '-', color='gray', zorder = 3)
        plt.fill_between(data_x, lower_err, upper_err,
                         color='gray', alpha=0.2, zorder = 3)
    elif errortype == 'DiscreteError':
        plt.errorbar(data_x, data_y,
                    yerr=sd_array,
                    fmt='-', capsize=5,  
                    elinewidth=2, markeredgewidth=2, zorder = 3)
    
    plt.xticks(xticks)
    plt.yticks(yticks)
    plt.xlabel(xlabel, fontsize=16, labelpad=10)
    plt.ylabel(ylabel, fontsize=16, labelpad=10)
    
    if mark:
        plt.plot((xticks[0], xticks[-1]), (1, 1), 'r-', zorder=0)
    
    plt.minorticks_off()
    
    plt.rcParams.update({'font.size': 12}) # This is for the size of axis
    plt.rcParams['axes.linewidth'] = 1.5 # set the value globally
    plt.rcParams['ytick.major.pad'] = 5 # Control the distance between axis and axis value
    plt.rcParams['xtick.major.pad'] = 10
    plt.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=True,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=True) # labels along the bottom edge are off
    plt.tick_params(
        axis='y',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        left=True,      # ticks along the bottom edge are off
        right=False)         # ticks along the top edge are off
    # here we modify the axes, specifying min and max for x and y axes.
    x_min_margin, x_max_margin, y_min_margin, y_max_margin = margin
    plt.axis(xmin=xticks[0] + x_min_margin, 
             xmax=xticks[-1] + x_max_margin, 
             ymin=yticks[0] + y_min_margin, 
             ymax=yticks[-1] + y_max_margin)
    plt.tight_layout()
    if save_fig == True:
        plt.savefig(fig_path, dpi=300)
    plt.show()
    
    
def plot_mulitplline_error(xticks, yticks, data_dic, xlabel, ylabel, margin, mark=True, save_fig=False, fig_path=None, errortype='ContinueError'):
#     plt.plot(data_x, data_y, 'or')
    # Same color as the seaborn
    colors = [(0.12156862745098039, 0.4666666666666667, 0.7058823529411765), (1.0, 0.4980392156862745, 0.054901960784313725), (0.17254901960784313, 0.6274509803921569, 0.17254901960784313)]
    ax_list = []
    label_list = []
    if errortype == 'ContinueError':
        num = 0
        for label in data_dic:
            label_list.append(label)
            data_temp = data_dic[label]
            data_x = data_temp['data_x'].values.tolist()
            data_y = data_temp['data_y'].values.tolist()
            lower_err = data_temp['lower_err'].values.tolist()
            upper_err = data_temp['upper_err'].values.tolist()
            ax_list.append(plt.plot(data_x, data_y, '-', zorder = 3, color=colors[num])[0])
            plt.fill_between(data_x, lower_err, upper_err,
                            alpha=0.2, zorder = 3, color=colors[num])
            num += 1
            
    elif errortype == 'DiscreteError':
        num = 0
        for label in data_dic:
            data_temp = data_dic[label]
            data_x = data_temp['data_x'].values.tolist()
            data_y = data_temp['data_y'].values.tolist()
            sd_array = data_temp['std_array'].values.tolist()
            label_list.append(label)
            ax_list.append(plt.plot(data_x, data_y, '-', zorder = 3, color=colors[num])[0])
            plt.errorbar(data_x, data_y,
                        yerr=sd_array,
                        fmt='-', capsize=5,  
                        elinewidth=2, markeredgewidth=2, zorder = 3, color=colors[num])
            num += 1
    
    plt.xticks(xticks)
    plt.yticks(yticks)
    plt.xlabel(xlabel, fontsize=16, labelpad=10)
    plt.ylabel(ylabel, fontsize=16, labelpad=10)
    plt.legend(ax_list, label_list)
    print(label_list)
    if mark:
        plt.plot((xticks[0], xticks[-1]), (1, 1), 'r-', zorder=0)
    
    plt.minorticks_off()
    
    plt.rcParams.update({'font.size': 12}) # This is for the size of axis
    plt.rcParams['axes.linewidth'] = 1.5 # set the value globally
    plt.rcParams['ytick.major.pad'] = 5 # Control the distance between axis and axis value
    plt.rcParams['xtick.major.pad'] = 10
    plt.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=True,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=True) # labels along the bottom edge are off
    plt.tick_params(
        axis='y',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        left=True,      # ticks along the bottom edge are off
        right=False)         # ticks along the top edge are off
    # here we modify the axes, specifying min and max for x and y axes.
    x_min_margin, x_max_margin, y_min_margin, y_max_margin = margin
    plt.axis(xmin=xticks[0] + x_min_margin, 
             xmax=xticks[-1] + x_max_margin, 
             ymin=yticks[0] + y_min_margin, 
             ymax=yticks[-1] + y_max_margin)
    plt.tight_layout()
    if save_fig == True:
        plt.savefig(fig_path, dpi=300)
    plt.show()