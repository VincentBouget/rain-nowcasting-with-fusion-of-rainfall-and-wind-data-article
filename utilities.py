# -*- coding: utf-8 -*-
# All rights reserved by Vincent Bouget, Arthur Filoche, Anastase Charantonis, Dominique Béréziat, Julien Brajard
# A research work funded by Sorbonne Center for Artificial Intelligence (Sorbonne Université)

import os
import torch
from data_viz import gen_plot, gen_plot_wind, gen_plot_rain

# To increment at each image saving
step_comparison = 1
step_batch = 1
step_U = 1
step_V = 1


def fetch_infos(filename):
    year,month,day,hour,minute = [k[1:] for k in filename.split(".")[0].split("-")]
    return int(year),int(month),int(day),int(hour),int(minute)


def batch_to_mapped_persistance(batch,thresholds):
    """
    Parameters
    ----------
    batch : Tensor BC(tempral)HW

    Returns
    -------
    persistance : Tensor BClsHW
        Keeps only the last image and stretch it in nb-of-classes.
    """
    imgs = batch
    persistance = torch.unsqueeze(map_to_classes(imgs[0,-1,:,:], thresholds,device=imgs.device),dim=0)
    for image in imgs[1:,-1,:,:]:
        mapped_pers = torch.unsqueeze(map_to_classes(image, thresholds,device=image.device),dim=0)
        persistance = torch.cat((persistance, mapped_pers),dim=0)  #BClsHW
    return persistance


def map_to_classes(img,threshold_list,device="cpu"):
    """
    Parameters
    ----------
    img : Torch Tensor HW or C(temporal)HW
    device : str, optional
        The default is "cpu".
    Returns
    -------
    result : Torch Tensor ClsWH or ClsCHW
    """
    nb_class = len(threshold_list) - 1 #the class [0,th1[ is set aside, no need to define a classifier for this one because its obvious that precipitation values must be >=0. 
    result = torch.zeros(tuple(nb_class if k==0 else img.shape[k-1] for k in range(len(img.shape)+1)),device=device)
    for i in range(nb_class):
        th1 = threshold_list[i+1]
        result[i] += (img>=th1)
    return result


def writer_add_batch_wind(wind_channels,text,writer,isU):
    """
    Parameters
    ----------
    imgs : Tensor Batch-VarChannel-X-Y
    text : str
        DESCRIPTION.
    writer : SummaryWriter
    """
    if isU:
        global step_U
    else:
        global step_V
    image = gen_plot_wind(wind_channels)
    writer.add_figure(text, image, step_batch)
    if isU:
        step_U += 1
    else:
        step_V += 1


def writer_add_batch_rain(rain_channels,text,writer):
    """
    Parameters
    ----------
    imgs : Tensor Batch-VarChannel-X-Y
    text : str
        DESCRIPTION.
    writer : SummaryWriter
    """
    global step_batch
    image = gen_plot_rain(rain_channels)
    writer.add_figure(text, image, step_batch)
    step_batch += 1
    
    
def writer_add_comparison(imgs,true_imgs,imgs_pred,text,writer,thresholds,temporal_length_inputs):
    global step_comparison
    # images_list=[]
    rain_channels = imgs[:, :temporal_length_inputs]
    persistance = batch_to_mapped_persistance(rain_channels,thresholds)
    for i in range(true_imgs.shape[0]):
        t_im=true_imgs[i]
        pred_i = (torch.sigmoid(imgs_pred[i]) > 0.5).float()
        pers_i = persistance[i]
        diff_pred = 2*pred_i-t_im
        diff_pers = 2*pers_i-t_im
        grouped_images = torch.stack((t_im,pred_i,pers_i,diff_pred,diff_pers),dim=0)     
        image = gen_plot(grouped_images)
        writer.add_figure(text,image,step_comparison)
        step_comparison += 1


def plot_comparison(imgs,true_imgs,imgs_pred,thresholds,temporal_length_inputs):
    rain_channels = imgs[:, :temporal_length_inputs]
    persistance = batch_to_mapped_persistance(rain_channels,thresholds)
    for i in range(true_imgs.shape[0]):
        t_im=true_imgs[i]
        pred_i = (torch.sigmoid(imgs_pred[i]) > 0.5).float()
        pers_i = persistance[i]
        diff_pred = 2*pred_i-t_im
        diff_pers = 2*pers_i-t_im
        grouped_images = torch.stack((t_im,pred_i,pers_i,diff_pred,diff_pers),dim=0)     
        image = gen_plot(grouped_images)
        return image
        


def list_leaves(root):
    """
    Parameters
    ----------
    root : str
        root directory.
    leaves_list : List
        empty.

    Returns
    -------
    leaves_list : List
        Pathes to files contained in the root.

    """
    if os.path.isfile(root):
        return([root])
    elif os.path.isdir(root):
        leaves_list=[]
        cc = os.listdir(root)
        for x in cc:
            x_path = os.path.join(root,x)
            leaves_list += list_leaves(x_path)
        return leaves_list
