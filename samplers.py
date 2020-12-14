# -*- coding: utf-8 -*-

import torch
from torch.utils.data.sampler import Sampler
from torch.utils.data import WeightedRandomSampler


class CustomSampler(Sampler):
    """
    Draws all element of indices one time and in the given order
    """
    
    def __init__(self, alist):
        """
        Parameters
        ----------
        alist : list
            Composed of True False for keep or reject position.
        """
        self.__alist___ = alist
        self.indices = [k for k in range(len(alist)) if alist[k]]

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)
   
    
def indices_except_undefined_sampler(dataset):
    """
    Parameters
    ----------
    dataset : MeteoNetDataset
    
    Returns
    -------
    samples_weight : list[True,False]
        A list of len(dataset) with True at position i if dataset[i]["target"] has a precipitation score > 0 and wind is defined.
    """
    PPMatrix = torch.load(dataset.PPMatrix)["data"]
    # We sum to obtain classes
    PPClass = torch.sum(PPMatrix,dim=1)
    U_Matrix = torch.load(dataset.U_Matrix)['definition']
    V_Matrix = torch.load(dataset.V_Matrix)['definition']
    assert ((len(PPClass) - dataset.temporal_length) // dataset.temporal_stride) == len(dataset)-1, f"Precipitation matrix size {len(PPClass)} does not match dataset size {len(dataset)}"
    assert ((len(U_Matrix) - dataset.temporal_length) // dataset.temporal_stride) == len(dataset)-1, f"Precipitation matrix size {len(U_Matrix)} does not match dataset size {len(dataset)}"
    assert ((len(V_Matrix) - dataset.temporal_length) // dataset.temporal_stride) == len(dataset)-1, f"Precipitation matrix size {len(V_Matrix)} does not match dataset size {len(dataset)}"
    samples_weight = []
    for i in range(len(dataset)-1):
        # transition imgs_dir to MeteoNetDataset : elt_i = ids[i*self.temporal_stride:self.temporal_stride*i + self.temporal_length]
        target_class = PPClass[dataset.temporal_stride*i + dataset.temporal_length]
        inputs_class = PPClass[dataset.temporal_stride*i: dataset.temporal_stride*i + dataset.temporal_length_inputs]
        inputs_and_target_windU_defined = U_Matrix[dataset.temporal_stride*i: dataset.temporal_stride*i + dataset.temporal_length]
        inputs_and_target_windV_defined = V_Matrix[dataset.temporal_stride*i: dataset.temporal_stride*i + dataset.temporal_length]
        conditions_to_meet = (target_class > 0) & (torch.sum(inputs_class<0)==0) & (torch.sum(inputs_and_target_windU_defined==False)==0) & (torch.sum(inputs_and_target_windV_defined==False)==0)
        samples_weight.append(conditions_to_meet.item())
    return samples_weight


def downsample_to_classes_above_classth_sampler(dataset, class_th):
    """
    Parameters
    ----------
    dataset : MeteoNetDataset
    class_th : int
        Class threshold. Condition : is >= class_th.

    Returns
    -------
    Tensor
        A tensor of length equals to len(dataset). Each element is 1 if the condition is met, 0 if the condition is not met
        and -1 if the element contains -1 (ie indefined elements)
    """
    PPMatrix = torch.load(dataset.PPMatrix)["data"]
    # Matrix is a vector (a,b,c) with 1 on index i if belongs to class i, we sum to obtain the class of each element.
    PPClass = torch.sum(PPMatrix,dim=1)
    U_Matrix = torch.load(dataset.U_Matrix)['definition']
    V_Matrix = torch.load(dataset.V_Matrix)['definition']
    assert ((len(PPClass) - dataset.temporal_length) // dataset.temporal_stride) == len(dataset)-1, f"Precipitation matrix size {len(PPClass)} does not match dataset size {len(dataset)}"
    assert ((len(U_Matrix) - dataset.temporal_length) // dataset.temporal_stride) == len(dataset)-1, f"Precipitation matrix size {len(U_Matrix)} does not match dataset size {len(dataset)}"
    assert ((len(V_Matrix) - dataset.temporal_length) // dataset.temporal_stride) == len(dataset)-1, f"Precipitation matrix size {len(V_Matrix)} does not match dataset size {len(dataset)}"
    samples_weight = []
    for i in range(len(dataset)-1):
        # transition rain_dir to MeteoNetDataset : elt_i = ids[i*self.temporal_stride:self.temporal_stride*i + self.temporal_length]
        target_class = PPClass[dataset.temporal_stride*i + dataset.temporal_length]
        # Condition to meet for targets
        condition_on_target_precipitation =  target_class >= class_th
        inputs_class = PPClass[dataset.temporal_stride*i: dataset.temporal_stride*i + dataset.temporal_length_inputs]
        inputs_and_target_windU_defined = U_Matrix[dataset.temporal_stride*i: dataset.temporal_stride*i + dataset.temporal_length]
        inputs_and_target_windV_defined = V_Matrix[dataset.temporal_stride*i: dataset.temporal_stride*i + dataset.temporal_length]
        # if one of the inputs or the target is not defined, the weight is -1
        if(target_class<0 or torch.sum(inputs_class<0)>0 or (torch.sum(inputs_and_target_windU_defined==False)>0) or (torch.sum(inputs_and_target_windV_defined==False)>0)):
            condition_on_target_precipitation = torch.tensor(-1)
        samples_weight.append(condition_on_target_precipitation.item())
    return torch.tensor(samples_weight)
    

def oversample_xpercent_rainy_tiles(dataset ,p , above_class):
    sp = downsample_to_classes_above_classth_sampler(dataset,above_class)
    # No oversampling
    if p == None:
        a = torch.ones(sp.shape)
        for i,elt in enumerate(sp):
            if elt==-1:
                a[i]=0 
        ts = WeightedRandomSampler(a, torch.sum(a==1).item(), replacement=True)
        return ts, (torch.sum(sp==1).item()/len(ts))
    else :
        # Number of sequences to exclude
        Excluded = torch.sum(sp==-1).item()
        N = len(sp) - Excluded
        # Sequences to oversample
        NR = int(torch.sum(sp==1).item())
        NN = N-NR
        # Probability to attribute to sequences to oversample
        pR = p/NR
        pN = (1-p)/NN
        a = (pR-pN)*sp+pN
        # Set proba to 0 for elt undefined
        for i,elt in enumerate(sp):
            if elt==-1:
                a[i]=0
        # Number of elements to draw
        newN = max( int((torch.sum(sp==0)/(1-p)).item()), int((torch.sum(sp==1)/p).item()) )
        ts = WeightedRandomSampler(a, newN, replacement=True)
        # Counts the percentage of oversampling after the drawing
        c = list(ts)
        s=0
        rainy_pos = torch.where(sp==1)[0]
        for i in c:
            if i in rainy_pos:
                s+=1
        
        return ts,(s/len(ts))