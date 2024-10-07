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
from scipy.spatial import distance
from scipy.ndimage import sobel
from scipy.ndimage import rotate


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


def find_nearest_skeleton_point(endpoint, skeleton_mask):
    
    skeleton_points = np.argwhere(skeleton_mask > 0)
    nearest_point = skeleton_points[distance.cdist([endpoint], skeleton_points).argmin()]

    return nearest_point


def calculate_angle_at_endpoint(point1, point2):
    delta_y = point2[-2] - point1[-2]
    delta_x = point2[-1] - point1[-1]
    angle_rad = np.arctan2(delta_y, delta_x)  
    angle_deg = np.degrees(angle_rad)

    if -90 < angle_deg < 0:
      angle_deg -= 90
    elif 0 < angle_deg < 90:
      angle_deg += 90
    elif -180 < angle_deg < -90:
      angle_deg += 90
    elif 90 < angle_deg < 180:
      angle_deg -= 90

    return angle_deg

def create_cone_kernel(size, intensity_center=1.0, angle_start=-25, angle_end=25):
    """
    Creates a cone kernel within a specific angular range.

    Parameters:
    - size: The size of the kernel (15 x 15 as the defult)
    - intensity_center: The maximum intensity value at the center.
    - angle_start: The starting angle of the cone (in degrees).
    - angle_end: The ending angle of the cone (in degrees).

    Returns:
    - cone_kernel: The generated cone kernel.
    """
    kernel = np.zeros((size, size), dtype=np.float32)

    center = ((size - 1) // 2, (size - 1) // 2)

    for i in range(size):
        for j in range(size):
            
            dist = np.sqrt((i - center[0]) ** 2 + (j - center[1]) ** 2)
            
            max_dist = np.sqrt(center[0] ** 2 + center[1] ** 2)
            normalized_dist = dist / max_dist

            angle = np.degrees(np.arctan2(i - center[0], j - center[1]))

            if angle_start <= angle <= angle_end:
                kernel[i, j] = intensity_center * (1 - normalized_dist)

    return kernel

def place_rotated_cone(endpoint, angle, kernel, mask):
    x, y = endpoint[-1], endpoint[-2]

    # print(f"Placing cone at endpoint ({x}, {y}) with angle {angle} degrees")

    rotated_kernel = rotate(kernel, angle, reshape=False, order=0)
    rotated_kernel /= rotated_kernel.sum()
    rotated_kernel = 10*rotated_kernel

    # # Debugging
    # plt.imshow(rotated_kernel, cmap='gray')
    # plt.title(f"Rotated Kernel at {angle} degrees")
    # plt.show()

    kernel_h, kernel_w = rotated_kernel.shape

    center_x = kernel_h // 2
    center_y = kernel_w // 2

    for i in range(center_x):
        for j in range(center_y):
            
            if -90 < angle <= 0:
              mask_x = x + i
              mask_y = y + j

              kernel_x = i + center_x
              kernel_y = j + center_y

            elif 0 < angle <= 90:
              mask_x = x + i
              mask_y = y - j

              kernel_x = i + center_x
              kernel_y = center_y - j

            elif -180 < angle <= -90:
              mask_x = x - i
              mask_y = y + j

              kernel_x = center_x - i
              kernel_y = j + center_y

            elif 90 < angle <= 180:
              mask_x = x - i
              mask_y = y - j

              kernel_x = center_x - i
              kernel_y = center_y - j

            else:
              
              raise ValueError(f"Unexpected angle value: {angle}")

            temp = rotated_kernel[kernel_y, kernel_x]
            if 0 <= mask_x < mask.shape[1] and 0 <= mask_y < mask.shape[2]:
              mask[:, mask_y, mask_x] += temp  

    return mask