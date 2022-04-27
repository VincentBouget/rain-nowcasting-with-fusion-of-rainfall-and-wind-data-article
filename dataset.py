# -*- coding: utf-8 -*-
# All right reserved by Vincent Bouget, Arthur Filoche, Anastase Charantonis, Dominique Béréziat, Julien Brajard
# A word funded by Sorbonne Center for Artificial Intelligence (Sorbonne Université)
import os
import torch
import logging 
import numpy as np
from utilities import fetch_infos
from torch.utils.data import Dataset
from utilities import map_to_classes


class MeteoNetDataset(Dataset):
    def __init__(self, rain_dir, U_dir, V_dir, temporal_length_inputs, temporal_length, temporal_stride ,thresholds, Matrix_path):
        """
        Parameters
        ----------
        imgs_list : List<str>
            Ordered (y[YEAR]-M[MONTH]-day[DAY]-h[HOUR]-m[MINUTE]) list of path to rain files
        wind_dir : str
            Path to wind directory
        temporal_length_inputs : int
            Length of the input sequence. Recommended is 12 for 1 hour.
        temporal_length : int
            Length of the sequence including inputs and the target, only the last one of the sequence is used as target. Recommended is 18 (12 for 1 hour input and prediction at +30min).
        temporal_stride : int
            Offset between each input sequence. Set to temporal_length_inputs if you dont want temporal overlapping.
        thresholds : List<float>
            List of thresholds to define classes
        """
        self.rain_dir = rain_dir
        files = [os.path.splitext(file)[0] for file in os.listdir(rain_dir) if os.path.isfile(os.path.join(rain_dir,file))]
        self.ids = sorted(files, key=lambda x: fetch_infos(x))
        self.U_dir = U_dir
        self.V_dir = V_dir
        self.temporal_length_inputs = temporal_length_inputs
        self.temporal_length = temporal_length
        self.temporal_stride = temporal_stride
        self.thresholds = thresholds
        
        # Import and checks Matrices
        self.PPMatrix = os.path.join(rain_dir, Matrix_path)
        assert os.path.exists(self.PPMatrix), f"Precipitation matrix path specified does not exist : {self.PPMatrix}"
        self.U_Matrix =  os.path.join(U_dir, Matrix_path) 
        assert os.path.exists(self.U_Matrix), f"Precipitation matrix path specified does not exist : {self.U_Matrix}"
        self.V_Matrix = os.path.join(V_dir, Matrix_path)
        assert os.path.exists(self.V_Matrix), f"Precipitation matrix path specified does not exist : {self.V_Matrix}"
        
        # Calculate normalization factors
        self.norm_factor = np.log( 1 + torch.load(self.PPMatrix)['max'] )
        self.norm_factor_U = {'mean': torch.load(self.U_Matrix)['mean'], 'variance': torch.load(self.U_Matrix)['variance']}
        self.norm_factor_V = {'mean': torch.load(self.V_Matrix)['mean'], 'variance': torch.load(self.V_Matrix)['variance']}
        
        logging.info('Creating dataset')

    def __len__(self):
        return 1 + (len(self.ids) - self.temporal_length) // self.temporal_stride
    
    def __getitem__(self, i):
        """
        Parameters
        ----------
        i : int
        
        Returns
        -------
        dict
            Inputs : C(temporal)HW.
            Target : ClsHW
        """
        
        assert i<len(self), f'Element index out of dataset bounds : {i} // 0-{len(self)-1}'
        idx = self.ids[i*self.temporal_stride:self.temporal_stride*i + self.temporal_length]
        imgs_files = [os.path.join(self.rain_dir,k)+'.npz' for k in idx]
        U_files = [os.path.join(self.U_dir,os.path.split(k)[-1]+'.npz') for k in idx]
        V_files = [os.path.join(self.V_dir,os.path.split(k)[-1]+'.npz') for k in idx]
        
        wind_exist = True
        for j in range(len(idx)):
            Ufile = U_files[j]
            Vfile = V_files[j]
            if not (os.path.isfile(Ufile) & os.path.isfile(Vfile)):
                wind_exist = False
                
        if wind_exist:                
            #inputs sequence
            assert len(imgs_files) == self.temporal_length, \
                f'Images found for the ID {idx}: {imgs_files} doesn\'t match {self.temporal_length}'
            rain_file = np.load(imgs_files[0])
            rain_data = list(rain_file.values())[0]
            sequence = torch.unsqueeze(torch.tensor(rain_data,dtype=torch.float32),dim=0)
            sequence = torch.log(1+sequence)/self.norm_factor
            #loads rainfall and wind
            for file in imgs_files[1:self.temporal_length_inputs]:
                rain_file = np.load(file)
                rain_data = list(rain_file.values())[0]
                img = torch.unsqueeze(torch.tensor(rain_data,dtype=torch.float32),dim=0)
                #Normalization:
                img = torch.log(1+img)/self.norm_factor
                sequence = torch.cat((sequence,img),dim=0)
            for file in U_files[:self.temporal_length_inputs]:
                img = torch.unsqueeze(torch.tensor(np.load(file)['data'],dtype=torch.float32),dim=0)
                #Normalization:
                img = (img-self.norm_factor_U['mean'])/np.sqrt(self.norm_factor_U['variance'])
                sequence = torch.cat((sequence,img),dim=0)
            for file in V_files[:self.temporal_length_inputs]:
                img = torch.unsqueeze(torch.tensor(np.load(file)['data'],dtype=torch.float32),dim=0)
                #Normalization:
                img = (img-self.norm_factor_V['mean'])/np.sqrt(self.norm_factor_V['variance'])
                sequence = torch.cat((sequence,img),dim=0)
            #target
            target_file = np.load(imgs_files[-1])
            target_data = list(target_file.values())[0]
            target = torch.from_numpy(target_data)
            return {'inputs': sequence, 'target': map_to_classes(target,self.thresholds), "identifier":idx[0]}
    
        else :
            #Wind is not defined for this image, will be ignore by sampler.
            return {'inputs': None, 'target': None, "identifier":idx[0]}
    

#%%

class DatasetPrediction(Dataset):
    def __init__(self, rain_dir, U_dir, V_dir, thresholds):
        self.rain_dir = rain_dir
        files = [os.path.splitext(file)[0] for file in os.listdir(rain_dir) if os.path.isfile(os.path.join(rain_dir,file))]
        self.ids = sorted(files, key=lambda x: fetch_infos(x))
        self.U_dir = U_dir
        self.V_dir = V_dir
        self.temporal_length_inputs = 12
        self.temporal_length = 18
        self.thresholds = thresholds
        
        # Normalization factors
        self.norm_factor = 8.9
        self.norm_factor_U = {'mean': 71, 'variance': 183038}
        self.norm_factor_V = {'mean': 19, 'variance': 175321}
        
        logging.info('Creating dataset')

    def __len__(self):
        return 1
    
    def __getitem__(self, i):
        assert i==0, f'Element index out of dataset bounds : {i} // 1'
        imgs_files = [os.path.join(self.rain_dir,k)+'.npz' for k in self.ids]
        U_files = [os.path.join(self.U_dir,os.path.split(k)[-1]+'.npz') for k in self.ids]
        V_files = [os.path.join(self.V_dir,os.path.split(k)[-1]+'.npz') for k in self.ids]
                
        #inputs sequence
        rain_file = np.load(imgs_files[0])
        rain_data = list(rain_file.values())[0]
        sequence = torch.unsqueeze(torch.tensor(rain_data,dtype=torch.float32),dim=0)
        sequence = torch.log(1+sequence)/self.norm_factor
        #loads rainfall and wind
        for file in imgs_files[1:self.temporal_length_inputs]:
            rain_file = np.load(file)
            rain_data = list(rain_file.values())[0]
            img = torch.unsqueeze(torch.tensor(rain_data,dtype=torch.float32),dim=0)
            #Normalization:
            img = torch.log(1+img)/self.norm_factor
            sequence = torch.cat((sequence,img),dim=0)
        for file in U_files[:self.temporal_length_inputs]:
            U_file = np.load(file)
            U_data = list(U_file.values())[0]
            img = torch.unsqueeze(torch.tensor(U_data,dtype=torch.float32),dim=0)
            #Normalization:
            img = (img-self.norm_factor_U['mean'])/np.sqrt(self.norm_factor_U['variance'])
            sequence = torch.cat((sequence,img),dim=0)
        for file in V_files[:self.temporal_length_inputs]:
            V_file = np.load(file)
            V_data = list(V_file.values())[0]
            img = torch.unsqueeze(torch.tensor(V_data,dtype=torch.float32),dim=0)
            #Normalization:
            img = (img-self.norm_factor_V['mean'])/np.sqrt(self.norm_factor_V['variance'])
            sequence = torch.cat((sequence,img),dim=0)
        #target
        target_file = np.load(imgs_files[-1])
        target_data = list(target_file.values())[0]
        target = torch.from_numpy(target_data)
        return {'inputs': sequence, 'target': map_to_classes(target,self.thresholds)}
