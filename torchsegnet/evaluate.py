import torch
import torch.nn.functional as F
from tqdm import tqdm

import torchsegnet as tn

# See: https://github.com/milesial/Pytorch-UNet/blob/master/evaluate.py
def evaluate(net, dataloader, device):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0
    loss = 0

    # iterate over the validation set
    for val_images, val_masks in tqdm(dataloader, total=num_val_batches, desc='Validation', unit='batch', leave=False):
        # move images and labels to correct device and type
        val_images = val_images[:,0:1,:,:].to(device=device, dtype=torch.float32)
        val_masks = val_masks.to(device=device, dtype=torch.float32)

        with torch.no_grad():
            # predict the mask
            pred_mask = net(val_images)
            
             # compute the Dice score
            dice_score += tn.dice_coeff(pred_mask, val_masks, reduce_batch_first=False)
            # compute the loss
            loss += (tn.BCE_loss(pred_mask, val_masks) + tn.dice_loss(pred_mask, val_masks))

    net.train()

    # Fixes a potential division by zero error
    if num_val_batches == 0:
        return dice_score, loss
    return (dice_score / num_val_batches), (loss/num_val_batches)
