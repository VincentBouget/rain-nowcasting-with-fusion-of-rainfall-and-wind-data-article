# -*- coding: utf-8 -*-

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
            sequence = torch.unsqueeze(torch.tensor(np.load(imgs_files[0])['data'],dtype=torch.float32),dim=0)
            sequence = torch.log(1+sequence)/self.norm_factor
            #loads rainfall and wind
            for file in imgs_files[1:self.temporal_length_inputs]:
                img = torch.unsqueeze(torch.tensor(np.load(file)['data'],dtype=torch.float32),dim=0)
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
            target = torch.from_numpy(np.load(imgs_files[-1])['data'])
            return {'inputs': sequence, 'target': map_to_classes(target,self.thresholds), "identifier":idx[0]}
    
        else :
            #Wind is not defined for this image, will be ignore by sampler.
            return {'inputs': None, 'target': None, "identifier":idx[0]}
    

#%% Used in : calculate_precipitation_matrix.py

class RainDataset(Dataset):
    def __init__(self, imgs_dir, spatial_length):
        """
        Parameters
        ----------
        imgs_dir : str
            Path to images directory.
        """
        self.imgs_dir = imgs_dir
        files = [os.path.splitext(file)[0] for file in os.listdir(imgs_dir) if os.path.isfile(os.path.join(imgs_dir,file))]
        self.ids = sorted(files, key=lambda x: (self.fetch_infos(x)))
        self.spatial_length = spatial_length

    def __len__(self):
        # (N-F)/S +1
        return len(self.ids)
    
    @classmethod
    def fetch_infos(cls,filename):
        year,month,day,hour,minute = [k[1:] for k in filename.split(".")[0].split("-")]
        return int(year),int(month),int(day),int(hour),int(minute)
    
    def __getitem__(self, i):
        """
        Parameters
        ----------
        i : int
        
        Returns
        -------
        Image i
        """
        assert i<len(self), \
            f'Element index out of dataset bounds : {i} // 0-{len(self)-1}'
        idx = self.ids[i]
        img_file = os.path.join(self.imgs_dir,idx+'.npz')
        #inputs sequence
        image = torch.from_numpy(np.load(img_file)['data'])
        assert (len(image.shape)==2 and image.shape[0]==self.spatial_length and image.shape[1]==self.spatial_length), f"image shape does not match : {image.shape}, should be ({self.spatial_length},{self.spatial_length})"
        return image
    
#%% Used in : calculate_wind_matrix.py

class WindDataset(Dataset):
    def __init__(self, rain_dir, wind_dir, spatial_length):
        """
        Parameters
        ----------
        rain_dir : str
            Path to rain directory.
        wind_dir : str
            Path to wind directory.
        """
        self.rain_dir = rain_dir
        self.wind_dir = wind_dir
        self.spatial_length = spatial_length
        files = [os.path.splitext(file)[0] for file in os.listdir(rain_dir) if os.path.isfile(os.path.join(rain_dir,file))]
        self.ids = sorted(files, key=lambda x: (self.fetch_infos(x)))

    def __len__(self):
        # (N-F)/S +1
        return len(self.ids)
    
    @classmethod
    def fetch_infos(cls,filename):
        year,month,day,hour,minute = [k[1:] for k in filename.split(".")[0].split("-")]
        return int(year),int(month),int(day),int(hour),int(minute)
    
    def __getitem__(self, i):
        """
        Parameters
        ----------
        i : int
        
        Returns
        -------
        Image i
        """
        assert i<len(self), \
            f'Element index out of dataset bounds : {i} // 0-{len(self)-1}'
        idx = self.ids[i]
        img_file = os.path.join(self.wind_dir,idx+'.npz')
        if os.path.isfile(img_file):
            #inputs sequence
            image = torch.from_numpy(np.load(img_file)['data']).to(dtype=torch.float32)
            assert (len(image.shape)==2 and image.shape[0]==self.spatial_length and image.shape[1]==self.spatial_length), f"image shape does not match : {image.shape}, should be ({self.spatial_length},{self.spatial_length})"
            return image
        else:
            return torch.tensor(0,dtype=torch.float32)