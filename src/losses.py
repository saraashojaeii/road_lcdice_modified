import numpy as np
import cv2
import torch.nn as nn
import torch
import torch.nn.functional as F
import os
import glob
from skimage.morphology import skeletonize
from torch.nn.modules.loss import *
from scipy.ndimage import distance_transform_edt
from unet_utils import *
from semseg_utils import *

def get_relaxed_precision(a, b, buffer):
    tp = 0
    indices = np.where(a == 1)
    for ind in range(len(indices[0])):
        tp += (np.sum(
            b[indices[0][ind]-buffer: indices[0][ind]+buffer+1,
              indices[1][ind]-buffer: indices[1][ind]+buffer+1]) > 0).astype(np.int32)
    return tp


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()
    
    def forward(self, y_pred, y_true):
        smooth = 1.
        # Use reshape to avoid issues with non-contiguous tensors
        y_true_f = y_true.reshape(-1)
        y_pred_f = y_pred.reshape(-1)
        intersection = (y_true_f * y_pred_f).sum()
        
        return 1 - ((2. * intersection + smooth) / (y_true_f.sum() + y_pred_f.sum() + smooth))


class BCEWithDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCELoss()

    def forward(self, y_pred, y_true):
        return self.bce(y_pred, y_true) + DiceLoss()(y_pred, y_true)


class LcDiceLoss(nn.Module):
    def __init__(self):
        super(LcDiceLoss, self).__init__()

    def lc_dice_loss(self, inputs, targets, alpha=1.0, beta=1.0):
        inputs = torch.sigmoid(inputs)  # Apply sigmoid to get probabilities
        
        # Ensure inputs and targets have the same shape
        if inputs.shape != targets.shape:
            targets = F.one_hot(targets.squeeze(1).long(), num_classes=inputs.shape[1])
            targets = targets.permute(0, 3, 1, 2).float()
        
        # Flatten the tensors using reshape
        inputs = inputs.reshape(inputs.size(0), -1)
        targets = targets.reshape(targets.size(0), -1)
        
        # Log-Cosh loss
        log_cosh = torch.log(torch.cosh(inputs - targets))
        log_cosh_loss = torch.mean(log_cosh)
        
        # Dice loss
        intersection = (inputs * targets).sum(dim=1)
        dice_loss = 1 - (2. * intersection + 1) / (inputs.sum(dim=1) + targets.sum(dim=1) + 1)
        dice_loss = dice_loss.mean()
        
        # lcDice loss
        lc_dice_loss = alpha * log_cosh_loss + beta * dice_loss
        return lc_dice_loss
    
    def forward(self, pred, target):
        
        lcd = self.lc_dice_loss(pred, target)
        
        return lcd


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


class GapLosswithL2(nn.Module):
    def __init__(self):
        super().__init__()
        self.l2 = nn.MSELoss()
        self.gap = GapLoss(K=1)

    def forward(self, y_pred, y_true):
        return self.l2(y_pred, y_true) + self.gap(y_pred, y_true)


class BCE_SAC_lcDice(nn.Module):
    def __init__(self, num_classes, alpha, beta, phi, cel, ftl, lcl, K=3):
        super(BCE_SAC_lcDice, self).__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.beta = beta
        self.phi = phi
        self.cel = cel
        self.ftl = ftl
        self.lcl = lcl
        self.K = K

    def lc_dice_loss(self, inputs, targets, alpha=1.0, beta=1.0):
        inputs = torch.sigmoid(inputs)  # Apply sigmoid to get probabilities
        
        # Ensure inputs and targets have the same shape
        if inputs.shape != targets.shape:
            targets = F.one_hot(targets.squeeze(1).long(), num_classes=inputs.shape[1])
            targets = targets.permute(0, 3, 1, 2).float()
        
        # Flatten the tensors using reshape
        inputs = inputs.reshape(inputs.size(0), -1)
        targets = targets.reshape(targets.size(0), -1)
        
        # Log-Cosh loss
        log_cosh = torch.log(torch.cosh(inputs - targets))
        log_cosh_loss = torch.mean(log_cosh)
        
        # Dice loss
        intersection = (inputs * targets).sum(dim=1)
        dice_loss = 1 - (2. * intersection + 1) / (inputs.sum(dim=1) + targets.sum(dim=1) + 1)
        dice_loss = dice_loss.mean()
        
        # lcDice loss
        lc_dice_loss = alpha * log_cosh_loss + beta * dice_loss
        return lc_dice_loss

    
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
        pred_class = torch.argmax(pred, dim=1)
        d = np.ones(self.num_classes)
        for c in range(self.num_classes):
            t = (target == c).sum()
            d[c] = t
        d = d / d.sum()
        d = 1 - d
        return torch.from_numpy(d).float()

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
        loss_seg = nn.CrossEntropyLoss(weight=self.weights(pred, target).cuda())
        ce_seg = loss_seg(pred, target_squeezed)
        
        pred_weighted = self.GapMat(pred, target)
        tv = self.tversky_loss(target, pred_weighted, alpha=self.alpha, beta=self.beta)
        
        # Ensure the target for lc_dice_loss is in the same shape as predictions
        lcd = self.lc_dice_loss(pred, target)
        
        total_loss = (self.cel * ce_seg) + (self.ftl * tv) + (self.lcl * lcd)
        return total_loss

