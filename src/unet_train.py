import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import os
import glob
from torch.utils.data import Dataset
from torchvision import transforms as T
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import time
from tqdm.notebook import tqdm
import wandb
import random

from semseg_utils import *
from unet_utils import *
from models.Unet_Network import *


data_path = '/root/home/MD/'
arg_dataset = 'deepglobe'

train_images, train_masks = data_pred(data_path, 'train', arg_dataset)
val_images, val_masks = data_pred(data_path, 'val', arg_dataset)

train_dataset = DataPrep(train_images, train_masks, transform=transform)
val_dataset = DataPrep(val_images, val_masks, transform=transform)

BATCH_SIZE = 4

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=False)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

unet_model = UNet()

optimizer = torch.optim.Adam(unet_model.parameters(), lr=0.001)
criterion = BCE_SAC(2, 1, 0.4, 4/3, 0.8, 0.2)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
unet_model.to(device)

EPOCHS = 100
t1 = time.time()
for epoch in range(EPOCHS):
    unet_model.train()  # Make sure to use 'unet_model' which is now correctly moved to the device

    total_train_loss = 0
    train_count = 0
        
    total_val_loss = 0
    val_count = 0
    val_average = 0
        

    for batch in tqdm(train_loader):
    # for batch in train_loader:
        inputs, target1 = batch
        inputs, target1 = inputs.to(device), target1.to(device)

        optimizer.zero_grad()

        mask_output = unet_model(inputs)  
        loss = criterion(mask_output, target1)

        total_train_loss += (loss.item()) 
        train_count += 1

        loss.backward()
        optimizer.step()

    train_average = total_train_loss / train_count   
    print(f"Epoch {epoch+1}/{EPOCHS}, Total Loss: {loss/len(train_loader)}, Mask Loss: {loss}")

    unet_model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for batch in tqdm(val_loader):
            inputs, target1 = batch
            inputs, target1 = inputs.to(device), target1.to(device)
            # target1 = target1.squeeze(1).long()
            
            mask_output = unet_model(inputs)
            
            val_loss = criterion(mask_output, target1)

            val_count += 1
            total_val_loss += val_loss.item()

            mask = mask_output.squeeze().cpu().numpy()
            val_y = target1.squeeze().cpu().numpy()  


    avg_val_loss = total_val_loss / val_count
    print(f"Validation: Epoch {epoch+1}/{EPOCHS}, Total Loss: {val_loss}, Mask Loss: {val_loss} ")

    val_average = total_val_loss / val_count

    print(f"Epoch: {(epoch+1)}, Training Loss: {train_average}, Validation Loss: {val_average}")
    

t2 = time.time()
print((t2 - t1))
