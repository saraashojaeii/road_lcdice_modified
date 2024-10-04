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

    # Calculate the center of the kernel
    center = ((size - 1) // 2, (size - 1) // 2)
    
    # Create a grid of distances and angles from the center
    for i in range(size):
        for j in range(size):
            # The Euclidean distance from the center
            dist = np.sqrt((i - center[0]) ** 2 + (j - center[1]) ** 2)
            # Normalize the distance 
            max_dist = np.sqrt(center[0] ** 2 + center[1] ** 2)
            normalized_dist = dist / max_dist

            # Compute the angle for this point relative to the center
            angle = np.degrees(np.arctan2(i - center[0], j - center[1]))
            
            # Apply the mask to create a wedge by only including points within the angular range
            if angle_start <= angle <= angle_end:
                # Compute the intensity as a function of distance (decreasing towards the edge)
                kernel[i, j] = intensity_center * (1 - normalized_dist)
            
    return kernel

def rotate_kernel(kernel, angle):
    h, w = kernel.shape
    center = (w // 2, h // 2)

    # Create the rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1)  # (center, angle, scale)

    # Perform the rotation
    rotated_kernel = cv2.warpAffine(kernel, rotation_matrix, (w, h), flags=cv2.INTER_LINEAR)

    return rotated_kernel


def find_nearest_skeleton_point(endpoint, skeleton_mask):

    # Get the coordinates of all skeleton points
    skeleton_points = np.argwhere(skeleton_mask > 0)
    # Find the nearest skeleton point to the endpoint
    nearest_point = skeleton_points[distance.cdist([endpoint], skeleton_points).argmin()]
    
    return nearest_point

def calculate_line_angle(point1, point2):
    delta_y = point2[1] - point1[1]
    delta_x = point2[0] - point1[0]
    angle_rad = np.arctan2(delta_y, delta_x)  # Use arctan2 for correct quadrant
    angle_deg = np.degrees(angle_rad)  # Convert to degrees
    return angle_deg