# -*- coding: utf-8 -*-
# All rights reserved by Vincent Bouget, Arthur Filoche, Anastase Charantonis, Dominique Béréziat, Julien Brajard
# A research work funded by Sorbonne Center for Artificial Intelligence (Sorbonne Université)

import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from utilities import batch_to_mapped_persistance


def eval_net_and_persistance(net, thresholds, loader, device):
    """Evaluation of the network on validation database"""
    
    net.eval()
    n_val = len(loader)  # number of batch
    epoch_loss = 0
    TP_FP_FN = 0
    TP_FP_FN_pers = 0
    Scores = {} 
    criterion = nn.BCEWithLogitsLoss()

    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for batch in loader:
            
            # Prepare images
            imgs, true_imgs = batch['inputs'], batch['target']
            imgs = imgs.to(device=device, dtype=torch.float32)
            true_imgs = true_imgs.to(device=device, dtype=torch.float32)
            persistance = batch_to_mapped_persistance(imgs,thresholds)
            
            # Network output
            with torch.no_grad():
                imgs_pred = net(imgs)
                
            loss =  criterion(imgs_pred, true_imgs)
            epoch_loss += loss.item()
            
            #Prediction
            pred = torch.sigmoid(imgs_pred)    #BClsHW
            pred = (pred > 0.5).float().to(device)
            
            for i in range(pred.shape[0]):
                # Micro by class
                TP_FP_FN += calculate_TP_FP_FN(pred[i],true_imgs[i])
                #Scores pers
                TP_FP_FN_pers += calculate_TP_FP_FN(persistance[i],true_imgs[i])
                
            pbar.update()
            
    # Micro by class
    dic_precision_micro = {}
    dic_recall_micro = {}
    micro_prec_by_class = TP_FP_FN[:,0]/(TP_FP_FN[:,0]+TP_FP_FN[:,1])
    micro_recall_by_class = TP_FP_FN[:,0]/(TP_FP_FN[:,0]+TP_FP_FN[:,2])
    for i in range(len(TP_FP_FN)):
        dic_precision_micro[f"Class_{i+1}"] = micro_prec_by_class[i].item()
        dic_recall_micro[f"Class_{i+1}"] = micro_recall_by_class[i].item()
    Scores["Precision_micro_by_class"] = dic_precision_micro
    Scores["Recall_micro_by_class"] = dic_recall_micro
    
    # Loss
    losses = {"lossTot" : epoch_loss}
    Scores["Loss_val"] = losses
    
    # Scores persistance
    dic_precision_micro_pers = {}
    dic_recall_micro_pers = {}
    micro_prec_by_class_pers = TP_FP_FN_pers[:,0]/(TP_FP_FN_pers[:,0]+TP_FP_FN_pers[:,1])
    micro_recall_by_class_pers = TP_FP_FN_pers[:,0]/(TP_FP_FN_pers[:,0]+TP_FP_FN_pers[:,2])
    for i in range(len(TP_FP_FN_pers)):
        dic_precision_micro_pers[f"Class_{i+1}"] = micro_prec_by_class_pers[i].item()
        dic_recall_micro_pers[f"Class_{i+1}"] = micro_recall_by_class_pers[i].item()
    Scores["Precision_Persistance_micro_by_class"] = dic_precision_micro_pers
    Scores["Recall_Persistance_micro_by_class"] = dic_recall_micro_pers
    
    # F1-Score Micro
    dic_F1_micro = {}
    for i in range(len(TP_FP_FN)):
        prec = dic_precision_micro[f"Class_{i+1}"]
        rec = dic_recall_micro[f"Class_{i+1}"]
        dic_F1_micro[f"Class_{i+1}"] = 2*prec*rec/(prec+rec) if (prec!=0 or rec!=0) else np.nan
    Scores["F1_micro"] = dic_F1_micro
    
    return Scores


def calculate_TP_FP_FN(y_pred,y_target):
    """
    Parameters
    ----------
    y_pred : tensor ClsHW
    y_target : tensor ClsHW

    Returns
    -------
    Default : Precision and recall macro.
    """
    nb_class = y_pred.shape[0]
    TP_FP_FN = torch.zeros(nb_class,3)   #(Cls,2)
    diff = 2*y_pred - y_target
    #TP
    TP_FP_FN[:,0] = torch.sum(diff==1,dim=(1,2))
    #FP
    TP_FP_FN[:,1] = torch.sum(diff==2,dim=(1,2))
    #FN
    TP_FP_FN[:,2] = torch.sum(diff==-1,dim=(1,2))
    return TP_FP_FN


