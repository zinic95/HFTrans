import math
import random

import torch
import numpy as np
from medpy.metric.binary import hd95
from torch.optim.lr_scheduler import _LRScheduler

def diceCoeff(preds, gts, labels, loss=False):
    pred = torch.zeros_like(gts)
    gt = torch.zeros_like(gts)
    for label in labels:
        if loss:
            pred = torch.add(pred, preds[:,label,:,:,:])
        else:
            pred[preds == label] = 1
        gt[gts == label] = 1
    if (torch.sum(gt) == 0.0):
        gt = 1 - gt
        pred = 1 - pred
        return 2 * torch.sum(gt * pred) / (torch.sum(pred) + torch.sum(gt) + 1e-6 )
    else:
        return 2 * torch.sum(gt * pred) / (torch.sum(pred) + torch.sum(gt) + 1e-6 )

def volumeSimillarity(preds, gts, labels):
    pred = torch.zeros_like(gts)
    gt = torch.zeros_like(gts)
    for label in labels:
        pred[preds == label] = 1
        gt[gts == label] = 1
    numerator = abs(torch.count_nonzero(gt) - torch.count_nonzero(pred))
    denominator = torch.count_nonzero(gt) + torch.count_nonzero(pred)
    if denominator > 0:
        return 1 - numerator / denominator
    else:
        return None
    
def hausdorff95(preds, gts, labels, voxelspacing=(1.0, 1.0, 1.0)):
    pred = torch.zeros_like(gts)
    gt = torch.zeros_like(gts)
    for label in labels:
        pred[preds == label] = 1
        gt[gts == label] = 1
    if torch.count_nonzero(gt) == 0:
        if torch.count_nonzero(pred) == 0:
            return 0
        else:
            return 373.13
    elif torch.count_nonzero(pred) == 0:
        return 373.13
    pred = torch.squeeze(pred)
    gt = torch.squeeze(gt)
    return hd95(pred.cpu().numpy(), gt.cpu().numpy(), voxelspacing)


def extract_patch(t1_img, t2_img, t1ce_img, flair_img, label, center, patch_size):
    # random patch  
    if np.random.rand() <= 2/3:
        patch_idx = [random.randint(0, img - patch) for img, patch in zip(t1_img.shape, patch_size)]
    # centered patch
    else:
        center = [s - p // 2 if c + p // 2 > s else c for c, p, s in zip(center, patch_size, label.shape)]
        center = [p // 2 if c - p // 2 < 0 else c for c, p in zip(center, patch_size)] 
        patch_idx = [c - p // 2 for c, p in zip(center, patch_size)]

    t1_patch = t1_img[patch_idx[0]:patch_idx[0] + patch_size[0], patch_idx[1]:patch_idx[1] + patch_size[1], patch_idx[2]:patch_idx[2] + patch_size[2]]
    t2_patch = t2_img[patch_idx[0]:patch_idx[0] + patch_size[0], patch_idx[1]:patch_idx[1] + patch_size[1], patch_idx[2]:patch_idx[2] + patch_size[2]]
    t1ce_patch = t1ce_img[patch_idx[0]:patch_idx[0] + patch_size[0], patch_idx[1]:patch_idx[1] + patch_size[1], patch_idx[2]:patch_idx[2] + patch_size[2]]
    flair_patch = flair_img[patch_idx[0]:patch_idx[0] + patch_size[0], patch_idx[1]:patch_idx[1] + patch_size[1], patch_idx[2]:patch_idx[2] + patch_size[2]]
    label_patch = label[patch_idx[0]:patch_idx[0] + patch_size[0], patch_idx[1]:patch_idx[1] + patch_size[1], patch_idx[2]:patch_idx[2] + patch_size[2]]

    t1_patch = np.expand_dims(t1_patch, axis=0)
    t2_patch = np.expand_dims(t2_patch, axis=0)
    t1ce_patch = np.expand_dims(t1ce_patch, axis=0)
    flair_patch = np.expand_dims(flair_patch, axis=0)

    return torch.FloatTensor(t1_patch), torch.FloatTensor(t2_patch), torch.FloatTensor(t1ce_patch), torch.FloatTensor(flair_patch), torch.LongTensor(label_patch)

def test_img(t1_img, t2_img, t1ce_img, flair_img, label):
    t1_img = np.expand_dims(t1_img, axis=0)
    t2_img = np.expand_dims(t2_img, axis=0)
    t1ce_img = np.expand_dims(t1ce_img, axis=0)
    flair_img = np.expand_dims(flair_img, axis=0)

    return torch.FloatTensor(t1_img), torch.FloatTensor(t2_img), torch.FloatTensor(t1ce_img), torch.FloatTensor(flair_img), torch.LongTensor(label)