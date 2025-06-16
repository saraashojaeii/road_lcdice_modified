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

class DataPrep(Dataset):
    def __init__(self, image_paths, label_paths, transform=None):
        self.image_paths = image_paths
        self.label1_paths = label_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load and transform image
        image = Image.open(self.image_paths[idx]).convert('RGB')
        mask = Image.open(self.label1_paths[idx]).convert('L')
        image = image.resize((512, 512))
        mask = mask.resize((512, 512))

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask

# Define a simple transform
transform = T.Compose([
    T.ToTensor(),  # Automatically converts PIL images to tensors and scales to [0, 1]
])

def data_pred(DATA_DIR, str='train', dataset='mass'):

    images = os.path.join(DATA_DIR, str)
    masks = os.path.join(DATA_DIR, str + '_labels')

    if dataset=='mass':
        image_paths = glob.glob(os.path.join(images, '*.tiff'))
        label_paths = glob.glob(os.path.join(masks, '*.tif'))

    elif dataset=='cityscale':
        image_paths = glob.glob(os.path.join(images, '*_sat.png'))
        label_paths = glob.glob(os.path.join(masks, '*_gt.png'))

    elif dataset=='deepglobe':
        image_paths = glob.glob(os.path.join(images, '*_sat.jpg'))
        label_paths = glob.glob(os.path.join(masks, '*_mask.png'))

    elif dataset=='equa':
        image_paths = glob.glob(os.path.join(images, '*.png'))
        label_paths = glob.glob(os.path.join(masks, '*.png'))

    elif dataset=='spacenet':
        image_paths = glob.glob(os.path.join(images, '*_rgb.png'))
        label_paths = glob.glob(os.path.join(masks, '*_gt.png'))

    image_paths.sort()
    label_paths.sort()
    return image_paths, label_paths


class BCE_Tversky(nn.Module):
    def __init__(self, num_classes, alpha, beta, phi, cel, ftl):
        super(BCE_Tversky, self).__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.beta = beta
        self.phi = phi
        self.cel = cel
        self.ftl = ftl

    def tversky_loss(self, true, logits, alpha, beta, eps=1e-7):
       
        num_classes = logits.shape[1]
        if num_classes == 1:
            true_1_hot = torch.eye(num_classes + 1).cuda()[true.long().squeeze(1)]
            true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
            true_1_hot_f = true_1_hot[:, 0:1, :, :]
            true_1_hot_s = true_1_hot[:, 1:2, :, :]
            true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
            pos_prob = torch.sigmoid(logits)
            neg_prob = 1 - pos_prob
            probas = torch.cat([pos_prob, neg_prob], dim=1)
        else:
            device = true.device  
            true = true.squeeze(1).long()
            true_1_hot = torch.eye(num_classes, device=device)[true.squeeze(1)]
            true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
            probas = F.softmax(logits, dim=1)
        true_1_hot = true_1_hot.type(logits.type())
        dims = (0,) + tuple(range(2, true.ndimension()))
        intersection = torch.sum(probas * true_1_hot, dims)
        fps = torch.sum(probas * (1 - true_1_hot), dims)
        fns = torch.sum((1 - probas) * true_1_hot, dims)
        num = intersection
        denom = intersection + (alpha * fps) + (beta * fns)
        tversky_loss = (num / (denom + eps)).mean()
        return (1 - tversky_loss)**self.phi
    
    def weights(self, pred, target, epsilon = 1e-6):
        pred_class = torch.argmax(pred, dim = 1)
        d = np.ones(self.num_classes)
    
        for c in range(self.num_classes):
            t = 0
            t = (target == c).sum()
            d[c] = t
            
        d = d/d.sum()
        d = 1 - d
        return torch.from_numpy(d).float()
    
    def forward(self, pred, target):
        if self.cel + self.ftl != 1:
            raise ValueError('Cross Entropy weight and Tversky weight should sum to 1')
        
        loss_seg = nn.CrossEntropyLoss(weight = self.weights(pred, target).cuda())
        target_squeezed = torch.squeeze(target, 1)
        target_squeezed = target_squeezed.long()
        ce_seg = loss_seg(pred, target_squeezed)
        tv = self.tversky_loss(target, pred, alpha=self.alpha, beta=self.beta)
        
        total_loss = (self.cel * ce_seg) + (self.ftl * tv)

        return total_loss

class GapLoss(nn.Module):
    def __init__(self, K=60):
        super(GapLoss, self).__init__()
        self.K = K

    def forward(self, pred, target):
        # Input is processed by softmax function to acquire cross-entropy map L
        criterion = nn.CrossEntropyLoss(reduction='none')
        
        # Squeeze target tensor to remove the channel dimension
        target = target.squeeze(1).long()
        
        L = criterion(pred, target)

        # Input is binarized to acquire image A
        A = torch.argmax(pred, dim=1)

        # Skeleton image B is obtained from A
        A_np = A.cpu().numpy()
        B = np.zeros_like(A_np)
        for n in range(A_np.shape[0]):
            temp = skeletonize(A_np[n])
            temp = np.where(temp == True, 1, 0)
            B[n] = temp
        B = torch.from_numpy(B).to(pred.device).double()
        B = torch.unsqueeze(B, dim=1)

        # Generate endpoint map C
        kernel = torch.ones((1, 1, 3, 3), dtype=torch.double).to(pred.device)
        kernel[0][0][1][1] = 0
        C = F.conv2d(B, weight=kernel, bias=None, stride=1, padding=1, dilation=1, groups=1)
        C = torch.mul(B, C)
        C = torch.where(C == 1, 1, 0).double()

        # Generate weight map W
        kernel = torch.ones((1, 1, 9, 9), dtype=torch.double).to(pred.device)
        N = F.conv2d(C, weight=kernel, bias=None, stride=1, padding=4, dilation=1, groups=1)
        N = N * self.K
        temp = torch.where(N == 0, 1, 0)
        W = N + temp

        loss = torch.mean(W * L)
        return loss

class BCE_SAC(nn.Module):
    def __init__(self, num_classes, alpha, beta, phi, cel, ftl, K=3):
        super(BCE_SAC, self).__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.beta = beta
        self.phi = phi
        self.cel = cel
        self.ftl = ftl
        self.K = K

    
    def tversky_loss(self, true, logits, alpha, beta, eps=1e-7):
        num_classes = logits.shape[1]
        if num_classes == 1:
            true_1_hot = torch.eye(num_classes + 1).cuda()[true.long().squeeze(1)]
            true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
            true_1_hot_f = true_1_hot[:, 0:1, :, :]
            true_1_hot_s = true_1_hot[:, 1:2, :, :]
            true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
            pos_prob = torch.sigmoid(logits)
            neg_prob = 1 - pos_prob
            probas = torch.cat([pos_prob, neg_prob], dim=1)
        else:
            device = true.device  
            true = true.squeeze(1).long()
            true_1_hot = torch.eye(num_classes, device=device)[true.squeeze(1)]
            true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
            probas = F.softmax(logits, dim=1)
        true_1_hot = true_1_hot.type(logits.type())
        dims = (0,) + tuple(range(2, true.ndimension()))
        intersection = torch.sum(probas * true_1_hot, dims)
        fps = torch.sum(probas * (1 - true_1_hot), dims)
        fns = torch.sum((1 - probas) * true_1_hot, dims)
        num = intersection
        denom = intersection + (alpha * fps) + (beta * fns)
        tversky_score = (num / (denom + eps)).mean()
        return (1 - tversky_score)**self.phi

    def weights(self, pred, target, epsilon=1e-6):
        # Ensure target is on the CPU for processing
        target = target.cpu().numpy()
        # Initialize weights as an array of ones
        d = np.ones(self.num_classes, dtype=np.float32)
        # Compute the frequency of each class
        for c in range(self.num_classes):
            t = (target == c).sum()
            d[c] = t
        # Normalize and invert the frequencies
        d = d / d.sum()
        d = 1 - d
        # Convert to tensor and ensure the correct device
        weights = torch.from_numpy(d).float().to(pred.device)
        return weights

    def GapMat(self, pred, target):
        criterion = nn.CrossEntropyLoss(reduction='none')
        target = target.squeeze(1).long()
        L = criterion(pred, target)
        A = torch.argmax(pred, dim=1)
    
        # Ensure the tensor is on the CPU before converting to numpy
        A_cpu = A.cpu().numpy()
        distance_transform = distance_transform_edt(A_cpu == 0)
        threshold = 10
        # Apply the threshold
        distance_transform[distance_transform > threshold] = threshold
        # Invert the distance transform
        distance_transform_inverted = ((threshold - distance_transform) / threshold)
        # Normalize for visualization
        A2 = cv2.normalize(distance_transform_inverted, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
        B = np.zeros_like(A_cpu)
        for n in range(A_cpu.shape[0]):
            temp = skeletonize(A_cpu[n])
            temp = np.where(temp == True, 1, 0)
            B[n] = temp
        B = torch.from_numpy(B).to(pred.device).double()
        B = torch.unsqueeze(B, dim=1)
    
        kernel = torch.ones((1, 1, 3, 3), dtype=torch.double).to(pred.device)
        kernel[0][0][1][1] = 0
    
        C = F.conv2d(B, weight=kernel, bias=None, stride=1, padding=1, dilation=1, groups=1)
        C = torch.mul(B, C)
        C = torch.where(C == 1, 1, 0).double()
    
        kernel = torch.ones((1, 1, 9, 9), dtype=torch.double).to(pred.device)
    
        N = F.conv2d(C, weight=kernel, bias=None, stride=1, padding=4, dilation=1, groups=1)
        N = N * self.K
    
        temp = torch.where(N == 0, 1, 0)
        W1 = N + temp
    
        W1[W1 == 1] = 0 
        A2 = A2 / 255.0
    
        A2_tensor = torch.tensor(A2, dtype=torch.double).to(pred.device)
        A2_tensor = torch.unsqueeze(A2_tensor, dim=0).unsqueeze(dim=0)
        W1 = W1.squeeze(0).squeeze(0)
        W2 = torch.mul(A2_tensor, W1)
    
        temp2 = torch.where(W2 == 0, 1, 0)
        W = W2 + temp2
    
        output = W * L
        loss = torch.mean(W * L)
        return output


    def forward(self, pred, target):
        # print(self.cel + self.ftl + self.lcl)
        # if (self.cel + self.ftl + self.lcl) != 1:
        #     raise ValueError('Cross Entropy weight and Tversky weight should sum to 1')
        
        target_squeezed = target.squeeze(1).long()
        weights = self.weights(pred, target)
        if weights.size(0) != self.num_classes:
            raise ValueError(f"Weight tensor size {weights.size(0)} does not match num_classes {self.num_classes}")
        loss_seg = nn.CrossEntropyLoss(weight=weights.cuda())

        ce_seg = loss_seg(pred, target_squeezed)
        
        pred_weighted = self.GapMat(pred, target)
        tv = self.tversky_loss(target, pred_weighted, alpha=self.alpha, beta=self.beta)
        
        
        total_loss = (self.cel * ce_seg) + (self.ftl * tv) 
        return total_loss

class GapLosswithL2(nn.Module):
    def __init__(self):
        super().__init__()
        self.l2 = nn.MSELoss()
        self.gap = GapLoss(K=1)

    def forward(self, y_pred, y_true):
        return self.l2(y_pred, y_true) + self.gap(y_pred, y_true)


class BCE(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCELoss()

    def forward(self, y_pred, y_true):
        return self.bce(y_pred, y_true) 

