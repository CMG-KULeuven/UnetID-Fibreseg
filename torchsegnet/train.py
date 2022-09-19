#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Author: Rui Guo (KU Leuven), rui.guo1@kuleuven.be
Date: July 2022
"""

import torch
import torchlib as tl
from torch import optim
import pylib as py
import torchsegnet as tn
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader, random_split

def train_net(Data,
              image_size,
              output_dir,
              val_percent: float = 0.2,
              epochs: int = 10,
              batch_size: int = 8,
              learning_rate: float = 0.0001,
              net_var: str = 'UnetID',
              save_checkpoint: bool = True,
              save_trainingmodel: bool = False,
              amp: bool = False 
              ):

    device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
    
    data_transform = transforms.Compose([transforms.ColorJitter(brightness=0.3, contrast=0.3),
                                        transforms.GaussianBlur(kernel_size=5, sigma=(0.7, 1.3)),
                                        transforms.ToTensor()])
    dataset = tl.CTImagesDataset(Data, transform=data_transform)
    
    # Split into train / validation partitions
    val_percent= 0.2
    num_val = int(len(dataset) * val_percent)
    num_train = len(dataset) - num_val
    train_set, val_set = random_split(dataset, [num_train, num_val], generator=torch.Generator().manual_seed(0))
    max_value = torch.max(train_set[0][0])
    if max_value > 1.0:
        print(f'The maximum values of the data is {max_value}, please nomalize it between 0-1!')

    # Create the dataloaders
    loader_args = dict(batch_size=batch_size, num_workers=4, pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, drop_last=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)
    
    # specify the model to use
    if net_var == 'UnetID':
        net = tn.UnetID()
    elif net_var == 'Unet':
        net = tn.Unet()
    net.to(device=device)
    
    # Set up the optimizer
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    global_step = 0
    
    # Training 
    for epoch in range(1, epochs+1):
        net.train()
        epoch_loss = 0
        with tqdm(total=int(num_train), desc=f'Epoch {epoch}/{epochs}', unit='iter') as pbar:
            for train_images, train_masks in train_loader:
                train_images = train_images[:,0:1,:,:].to(device=device, dtype=torch.float32)
                train_masks = train_masks.to(device=device, dtype=torch.float32)
                
                with torch.cuda.amp.autocast(enabled=amp):
                    pred_masks = net(train_images)
                    loss = tn.BCE_loss(pred_masks, train_masks) \
                           + tn.dice_loss(pred_masks, train_masks)
                           
                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                optimizer.step()
                grad_scaler.update()
                
                pbar.update(train_images.shape[0])
                global_step += 1
                epoch_loss += loss
                pbar.set_postfix(**{'training loss (batch)': loss.item()})
        
        if epoch== 1 or epoch % 50 == 0:
            _, val_loss = tn.evaluate(net, val_loader, device)
            print(f'At the epoch {epoch}, validation loss is {val_loss.item()}')
            if save_checkpoint:
                dir_checkpoint = py.join(output_dir, 'checkpoint')
                py.mkdir(dir_checkpoint)
                torch.save(net.state_dict(), str(dir_checkpoint + '/checkpoint_epoch{}.pth'.format(epoch)))
                
        if epoch== 1 or epoch % 100 == 0:
            if save_trainingmodel:
                dir_model = py.join(output_dir, 'model')
                py.mkdir(dir_model)
                torch.save(net, str(dir_model + '/model_epoch{}.pth'.format(epoch)))