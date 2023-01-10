#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Author: Rui Guo (KU Leuven), rui.guo1@kuleuven.be
Date: July 2022
"""

import torch
from tqdm import tqdm
import time as time
from fibresegt.models import *
from fibresegt.data import *
from .evaluate import evaluate_net

def train_net(Data,
              image_size,
              output_dir,
              val_percent: float = 0.0,
              epochs: int = 10,
              batch_size: int = 8,
              learning_rate: float = 0.0001,
              net_var: str = 'UnetID',
              optimizer_method: str = "Adam",
              loss_method: str = "BCE+dice",
              data_aug: dict = None,
              preprocess_info: dict = None):
    
    # Set device
    device = get_device()
    
    # Load data
    (train_loader, num_train), (val_loader, num_val) = get_dataloaders(Data, 
                                                                       val_percent, 
                                                                       batch_size, 
                                                                       data_aug)
    
    # Load the neural network
    net = get_net(net_var).to(device)
    
    # Get loss function and optimizer
    optimizer = get_optimizer(optimizer_method, net.parameters(), learning_rate)
    amp = False
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    global_step = 0
    
    # Training
    start_time = time.time()
    for epoch in range(1, epochs+1):
        net.train()
        epoch_loss = 0
        with tqdm(total=int(num_train), desc=f'Epoch {epoch}/{epochs}', unit='iter') as pbar:
            for train_images, train_masks in train_loader:
                train_images = train_images[:,0:1,:,:].to(device=device, dtype=torch.float32)
                train_masks = train_masks.to(device=device, dtype=torch.float32)
                
                with torch.cuda.amp.autocast(enabled=amp):
                    pred_masks = net(train_images)
                    # if loss_method == "BCE+dice":
                    loss = BCE_loss(pred_masks, train_masks) \
                           + dice_loss(pred_masks, train_masks)
                           
                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                optimizer.step()
                grad_scaler.update()
                
                pbar.update(train_images.shape[0])
                global_step += 1
                epoch_loss += loss
                pbar.set_postfix(**{'training loss (batch)': loss.item()})
        
        if epoch == 1 or epoch % 100 == 0:
            if val_percent !=0.0 :
                _, val_loss = evaluate_net(net, val_loader, device)
                print(f'At the epoch {epoch}, validation loss is {val_loss.item()}')
                
            # Save the trained checkpoint to the specified directory.  
            dir_checkpoint = join(output_dir, 'checkpoint')
            mkdir(dir_checkpoint)
            checkpoint_path = dir_checkpoint + '/checkpoint_epoch{}.pth'.format(epoch)
            save_checkpoint(checkpoint_path=checkpoint_path, model=net)
                
    end_time = time.time()
    cost = end_time-start_time
    print('The time needed to train is: ', cost)
    
   # Save information
    hyperpara_info = dict(val_percent = val_percent,
                          epochs = epochs,
                          batch_size = batch_size,
                          learning_rate = learning_rate, 
                          net_var = net_var, 
                          optimizer_method = optimizer_method,
                          loss_method = loss_method,
                          data_aug = data_aug)
    saved_info = {
        "The training data information: ": preprocess_info,
        "The hyperparameters of the network: ": hyperpara_info,
        "The time needed to train is: ": cost
        }
    path = join(output_dir, 'NetworkPara')
    mkdir(path)
    write_to_json(path, saved_info)
