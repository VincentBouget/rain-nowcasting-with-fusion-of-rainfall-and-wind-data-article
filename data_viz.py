# -*- coding: utf-8 -*-
# All rights reserved by Vincent Bouget, Arthur Filoche, Anastase Charantonis, Dominique Béréziat, Julien Brajard
# A word funded by Sorbonne Center for Artificial Intelligence (Sorbonne Université)

from matplotlib import colors
import matplotlib.pyplot as plt


def gen_plot(inputs):
    """
    Parameters
    ----------
    inputs : Torch Tensor BCHW within [0,1]
        MeteoNet data are greyscale so channel input is used as temporal channel to store images of the 12-sequence. 

    Returns
    -------
    buf : Buffer
        Buffer to temporarily save the figure.

    """
    nb_batch = inputs.shape[0]
    nb_channels = inputs.shape[1]
    cmap_BW = colors.ListedColormap(['red','black', 'white'])
    cmap_col = colors.ListedColormap(['indianred', 'mediumseagreen' ,'green', 'red'])
    legend = "-1-Pale red : pred=0,target=1 / 0-SeaGreen : pred=0,target=0 / 1-Green : pred=1,target=1 / 2-Red : pred=1,target=0"
    fig,axes = plt.subplots(nb_batch,nb_channels,figsize=(15,nb_batch*2))
    n=1
    label = ["Target","Prediction","Persistance","2*Pred-Tgt","2*Pers-Tgt"]
    
    for i in range(nb_batch):
        for j in range(nb_channels):
            cmap=cmap_BW if (i<3) else cmap_col
            bounds =  [-1,-0.5,0.5,1] if (i<3) else [-1,-0.5,0.5,1.5,2]
            norm = colors.BoundaryNorm(bounds, cmap.N)
            ax = axes[i,j] if nb_batch>1 else axes[j]
            ax.set(aspect='equal')
            pl=ax.pcolormesh(inputs[i,j].to("cpu"),cmap=cmap, norm=norm)
            if j==0:
                ax.set_ylabel(label[i])
            n += 1
                
    cbar = fig.colorbar(pl,ax=axes.ravel().tolist(),cmap=cmap, norm=norm, boundaries=bounds, ticks=bounds, 
                    orientation= 'horizontal').set_label(legend)
    return fig


def gen_plot_rain(inputs):
    nb_batch = inputs.shape[0]
    nb_channels = inputs.shape[1]
    
    bounds = []
    legend = ""
    cmap = colors.ListedColormap(['black','white', 'darkslateblue','dodgerblue','skyblue','mediumseagreen','cyan','lime','yellow',
                                  'burlywood','orange','red'])
    legend = 'Rainfall / -1 : missing values' #(in 1/100 mm)
    bounds = [-1]+[k/100 for k in range(0,110,10)]
    norm = colors.BoundaryNorm(bounds, cmap.N)
    fig,axes = plt.subplots(nb_batch,nb_channels,figsize=(15,nb_batch*2))
    n=1
    for i in range(nb_batch):
        for j in range(nb_channels):
            ax = axes[i,j] if nb_batch>1 else axes[j]
            ax.set(aspect='equal')
            pl=ax.pcolormesh(inputs[i,j].to("cpu"),cmap=cmap, norm=norm) 
            n += 1
    
    cbar = fig.colorbar(pl,ax=axes.ravel().tolist(),cmap=cmap, norm=norm, boundaries=bounds, ticks=bounds, 
                    orientation= 'horizontal').set_label(legend)
    return fig

    
def gen_plot_wind(inputs):
    nb_batch = inputs.shape[0]
    nb_channels = inputs.shape[1]
    
    bounds = []
    legend = ""
    cmap = colors.ListedColormap(['black','white', 'darkslateblue','dodgerblue','skyblue','mediumseagreen','cyan','lime','yellow',
                                  'burlywood','orange','red'])
    legend = 'Wind' #(in 1/100 mm)
    bounds = [k/100 for k in range(-400,0,40)]+[k/100 for k in range(0,410,40)]
    norm = colors.BoundaryNorm(bounds, len(bounds)-1)
    fig,axes = plt.subplots(nb_batch,nb_channels,figsize=(15,nb_batch*2))
    n=1
    for i in range(nb_batch):
        for j in range(nb_channels):
            ax = axes[i,j] if nb_batch>1 else axes[j]
            ax.set(aspect='equal')
            pl=ax.pcolormesh(inputs[i,j].to("cpu"),cmap=plt.get_cmap('seismic'), vmin=-4, vmax=4) 
            n += 1
    
    cbar = fig.colorbar(pl,ax=axes.ravel().tolist(),cmap=cmap, norm=norm, boundaries=bounds, ticks=bounds, 
                    orientation= 'horizontal').set_label(legend)
    return fig
