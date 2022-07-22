import os
import sys
import logging
from tqdm import tqdm

import torch
import numpy as np
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import MeteoNetDataset
import samplers

from unet_model import UNet

from eval import eval_net_and_persistance
import utilities as utils

def train(net,
          rain_dir,
          U_dir,
          V_dir,
          ckp_dir,
          device,
          epochs=20,
          batch_size=256,
          lr={0:1e-4},
          save_cp=True,
          num_workers=0,
          n_val_round_by_epoch=3,
          wd={0:1e-5},
          percentage_sampling=0.8):

    train = MeteoNetDataset(rain_dir=os.path.join(rain_dir,'train'), 
                            U_dir=os.path.join(U_dir,'train'), 
                            V_dir=os.path.join(V_dir,'train'),
                            temporal_length_inputs=temporal_length_inputs, 
                            temporal_length=temporal_length, 
                            temporal_stride=temporal_length_inputs, 
                            thresholds=thresholds_in_cent_mm, 
                            Matrix_path=Matrix_path)
    val = MeteoNetDataset(rain_dir=os.path.join(rain_dir,'val'), 
                          U_dir=os.path.join(U_dir,'val'), 
                          V_dir=os.path.join(V_dir,'val'),
                          temporal_length_inputs=temporal_length_inputs, 
                          temporal_length=temporal_length, 
                          temporal_stride=temporal_length_inputs, 
                          thresholds=thresholds_in_cent_mm, 
                          Matrix_path=Matrix_path)
    thresholds_normalized = np.log(np.array(thresholds_in_cent_mm)+1)/train.norm_factor
    
    train_sampler, real_percentage = samplers.oversample_xpercent_rainy_tiles(train, p=percentage_sampling, 
                                                                              above_class=len(thresholds_in_cent_mm)-1)
    train_loader = DataLoader(train, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers, pin_memory=True)
    # Custom sampler to consider only defined data
    val_sampler = samplers.CustomSampler(samplers.indices_except_undefined_sampler(val))
    val_loader = DataLoader(val, batch_size=batch_size, sampler=val_sampler, num_workers=num_workers, pin_memory=True, drop_last=True)
    
    writer = SummaryWriter(comment=f'-LR_{lr}_BS_{batch_size}_E_{epochs}_T{temporal_length}_WD_{wd}_p{percentage_sampling}')
    writer.add_scalar('Number_of_parameters', net.get_nb_params())
    info = f'''Starting training:
        Epochs:                {epochs}
        Learning rate:         {lr}
        Batch size:            {batch_size}
        Weight decay:          {wd}
        Number batch train :   {len(train_loader)}
        Number batch val :     {len(val_loader)}
        Training size:         {len(train)}
        Validation size:       {len(val)}
        Checkpoints:           {save_cp}
        Device:                {device.type}
        Number of parameters:  {net.get_nb_params()}
        Thresholds(mm/h):      {thresholds_in_mmh}
        Temporal length:       {temporal_length}
        Normalization factor:  {train.norm_factor}
        Percentage sampling:   {percentage_sampling}
        Real percantage :      {real_percentage}
    '''
    logging.info(info)
    writer.add_text('Description', info)

    global_step = 0
    
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        
        if epoch in lr.keys():
            print('***:',lr[epoch], wd[epoch])
            optimizer = optim.Adam(net.parameters(), lr=lr[epoch], weight_decay=wd[epoch])
            
        net.train()
        epoch_loss = 0
        n_batch = len(train_loader)
        
        with tqdm(total=n_batch, desc=f'Epoch {epoch + 1}/{epochs}', unit='batch') as pbar:
            for batch in train_loader:
                imgs = batch['inputs']  #BC(temp)HW
                true_imgs = batch['target']   #BClsHW
                assert imgs.shape[1] == net.n_channels, \
                    f'Network has been defined with {net.n_channels} input channels, ' \
                    f'but loaded images have {imgs.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'
                   
                imgs = imgs.to(device=device, dtype=torch.float32)
                true_imgs = true_imgs.to(device=device, dtype=torch.float32)
                
                imgs_pred = net(imgs)   #BClsHW
                loss = criterion(imgs_pred, true_imgs)
                epoch_loss += loss.item()
                pbar.set_postfix(**{'loss (epoch)': epoch_loss})
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(net.parameters(), 0.1)
                optimizer.step()
                pbar.update()
                
                if ((global_step % (n_batch//n_val_round_by_epoch))==0):
                    Scores = eval_net_and_persistance(net=net, thresholds=thresholds_normalized, loader=val_loader, device=device)
                    # Log scores
                    for key in Scores.keys():
                        score = Scores[key]
                        if type(score)==type(dict()):
                            writer.add_scalars(f'{key}_val', score, global_step)
                        else :
                            writer.add_scalar(f'{key}_val', score, global_step)
                            
                    # Save images of batch inputs//target, prediction, persistance, (pred-target) and (pers-target) on the last epoch
                    if (epoch==epochs-1):    
                        # we select rain, U and V channels (BCHW)
                        utils.writer_add_batch_rain(rain_channels=imgs[:,:12,:,:],text="Rain_inputs",writer=writer)
                        utils.writer_add_batch_wind(wind_channels=imgs[:,12:24,:,:],text="U_inputs",writer=writer,isU=True)
                        utils.writer_add_batch_wind(wind_channels=imgs[:,24:,:,:],text="V_inputs",writer=writer,isU=False)
                        utils.writer_add_comparison(imgs,true_imgs,imgs_pred,
                                                    text="Target_Prediction_Persistance_Prediction-Target_Persistance-Target",
                                                    writer=writer,thresholds=thresholds_normalized, temporal_length_inputs=temporal_length_inputs)
        
                global_step += 1
        losses = {"lossTot" : epoch_loss}
        writer.add_scalars('Loss_train', losses, epoch)
        
        if save_cp:
            try:
                os.mkdir(ckp_dir)
                logging.info('Created checkpoint directory')
            except OSError:
                pass
            torch.save(net.state_dict(),
                       os.path.join(ckp_dir, f'CP_epoch{epoch + 1}.pth'))
            logging.info(f'Checkpoint {epoch + 1} saved !')
    writer.close()
