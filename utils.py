import numpy as np
import cv2
import torch.nn as nn
import torch
import torch.nn.functional as F
import os
import glob
from skimage.morphology import skeletonize
from torch.nn.modules.loss import *


def data_pred(DATA_DIR, str='train'):

    images = os.path.join(DATA_DIR, str)
    masks = os.path.join(DATA_DIR, str + '_labels')

    image_paths = glob.glob(os.path.join(images, '*.tiff'))
    label_paths = glob.glob(os.path.join(masks, '*.tif'))

    image_paths.sort()
    label_paths.sort()
    return image_paths, label_paths


def mIoU(pred, target, num_classes):
    # Ensure target is a NumPy array (in case it's not already)
    target = np.array(target)
    # Initialize IoU array
    iou = np.ones(num_classes)
    for c in range(num_classes):
        # Find pixels belonging to class c
        p = (pred == c)
        t = (target == c)
        
        if p.shape != t.shape:
            raise ValueError(f"Shape mismatch: pred shape {p.shape} does not match target shape {t.shape}")
        
        # Calculate intersection and union
        inter = np.float64((p & t).sum())  # Use logical AND for intersection
        union = p.sum() + t.sum() - inter
        
        # Avoid division by zero
        iou[c] = (inter + 0.001) / (union + 0.001)

    # Calculate mean IoU
    miou = np.mean(iou)
    return miou, iou

def relaxed_f1(pred, gt, buffer):
    ''' Usage and Call
    # rp_tp, rr_tp, pred_p, gt_p = relaxed_f1(predicted.cpu().numpy(), labels.cpu().numpy(), buffer = 3)

    # rprecision_tp += rp_tp
    # rrecall_tp += rr_tp
    # pred_positive += pred_p
    # gt_positive += gt_p

    # precision = rprecision_tp/(gt_positive + 1e-12)
    # recall = rrecall_tp/(gt_positive + 1e-12)
    # f1measure = 2*precision*recall/(precision + recall + 1e-12)
    # iou = precision*recall/(precision+recall-(precision*recall) + 1e-12)
    '''

    rprecision_tp, rrecall_tp, pred_positive, gt_positive = 0, 0, 0, 0
    # for b in range(pred.shape[0]):
    pred_sk = skeletonize(pred)
    gt_sk = skeletonize(gt)
        # pred_sk = pred[b]
    # gt_sk = gt[b]

    #The correctness represents the percentage of correctly extracted road data, i.e., the percentage
    #of the extracted data which lie within the buffer around the reference network (groudn truth):
    rprecision_tp += get_relaxed_precision(pred_sk, gt_sk, buffer)

    #The completeness is the percentage of the reference data which is explained by the extracted
    #data, i.e., the percentage of the reference network which lie within the buffer around the
    #extracted data (prediction):
    rrecall_tp += get_relaxed_precision(gt_sk, pred_sk, buffer)
    pred_positive += len(np.where(pred_sk == 1)[0])
    gt_positive += len(np.where(gt_sk == 1)[0])

    #Correctness corresponds to relaxed precision
    #Completeness corresponds to relaxed recall 
    #Quality corresponds to intersection-over-union

    comm= rrecall_tp/(gt_positive + 1e-12) #length of matched reference/ length of reference
    corr= rprecision_tp/(pred_positive + 1e-12)   #length of matched extraction/ length of extraction
    qul = (comm*corr )/(comm- (comm*corr) + corr+ 1e-12)
    return comm*100, corr*100, qul*100


def dice_loss(score, target):
    target = target.float()
    smooth = 1e-5
    intersect = torch.sum(score * target)
    y_sum = torch.sum(target * target)
    z_sum = torch.sum(score * score)
    score = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    loss = 1 - score
    return loss, score


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


class AdaptiveTverskyCrossEntropyWeightedLoss(nn.Module):
    def __init__(self, num_classes, alpha, beta, phi, cel, ftl, K=3):
        super(AdaptiveTverskyCrossEntropyWeightedLoss, self).__init__()
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
        A_np = A.cpu().numpy()
        B = np.zeros_like(A_np)
        for n in range(A_np.shape[0]):
            temp = skeletonize(A_np[n])
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
        W = N + temp
        output = W * L
        loss = torch.mean(W * L)
        return output

    def forward(self, pred, target):
        if self.cel + self.ftl != 1:
            raise ValueError('Cross Entropy weight and Tversky weight should sum to 1')
        
        target_squeezed = target.squeeze(1).long()
        loss_seg = nn.CrossEntropyLoss(weight=self.weights(pred, target).cuda())
        ce_seg = loss_seg(pred, target_squeezed)
        
        pred_weighted = self.GapMat(pred, target)
        tv = self.tversky_loss(target, pred_weighted, alpha=self.alpha, beta=self.beta)
        
        total_loss = (self.cel * ce_seg) + (self.ftl * tv)
        return total_loss

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


class AdaptiveTverskyCrossEntropyLcDiceWeightedLoss(nn.Module):
    def __init__(self, num_classes, alpha, beta, phi, cel, ftl, lcl, K=3):
        super(AdaptiveTverskyCrossEntropyLcDiceWeightedLoss, self).__init__()
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
        A_np = A.cpu().numpy()
        B = np.zeros_like(A_np)
        for n in range(A_np.shape[0]):
            temp = skeletonize(A_np[n])
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
        W = N + temp
        output = W * L
        loss = torch.mean(W * L)
        return output

    def forward(self, pred, target):
        if self.cel + self.ftl + self.lcl != 1:
            raise ValueError('Cross Entropy weight and Tversky weight should sum to 1')
        
        target_squeezed = target.squeeze(1).long()
        loss_seg = nn.CrossEntropyLoss(weight=self.weights(pred, target).cuda())
        ce_seg = loss_seg(pred, target_squeezed)
        
        pred_weighted = self.GapMat(pred, target)
        tv = self.tversky_loss(target, pred_weighted, alpha=self.alpha, beta=self.beta)
        
        # Ensure the target for lc_dice_loss is in the same shape as predictions
        lcd = self.lc_dice_loss(pred, target)
        
        total_loss = (self.cel * ce_seg) + (self.ftl * tv) + (self.lcl * lcd)
        return total_loss


class TverskyCrossEntropyDiceWeightedLoss(nn.Module):
    def __init__(self, num_classes, alpha, beta, phi, cel, ftl):
        super(TverskyCrossEntropyDiceWeightedLoss, self).__init__()
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
        output = W * L
        loss = torch.mean(W * L)
        return output


class GapLosswithL2(nn.Module):
    def __init__(self):
        super().__init__()
        self.l2 = nn.MSELoss()

    def forward(self, y_pred, y_true):
        return self.l2(y_pred, y_true) + GapLoss(K=1)(y_pred, y_true)

def get_relaxed_precision(a, b, buffer):
    tp = 0
    indices = np.where(a == 1)
    for ind in range(len(indices[0])):
        tp += (np.sum(
            b[indices[0][ind]-buffer: indices[0][ind]+buffer+1,
              indices[1][ind]-buffer: indices[1][ind]+buffer+1]) > 0).astype(np.int32)
    return tp


