import os
import math
import time
import numpy as np
import nibabel as nib
import h5py
import torch
import torchio as tio
import torch.nn.functional as F
from torchio import RandomElasticDeformation

def preprocessingBraTS(data_path, extension):
    image_path = data_path[0]
    image_name = data_path[1]
    img_t1  = nib.load(os.path.join(image_path, image_name+'_t1.'+extension))
    img_t2  = nib.load(os.path.join(image_path, image_name+'_t2.'+extension))
    img_t1ce  = nib.load(os.path.join(image_path, image_name+'_t1ce.'+extension))
    img_flair  = nib.load(os.path.join(image_path, image_name+'_flair.'+extension))
    img_seg  = nib.load(os.path.join(image_path, image_name+'_seg.'+extension))
    
    t1_numpy = img_t1.get_fdata()
    idx = [int(np.mean(array)) for array in np.nonzero(t1_numpy)]
    
    target = [192,224,160]
    t1_numpy = resizing(img_t1, idx, target)
    t2_numpy = resizing(img_t2, idx, target)
    t1ce_numpy = resizing(img_t1ce, idx, target)
    flair_numpy = resizing(img_flair, idx, target)
    seg_numpy = resizing(img_seg, idx, target)

    #canonical_img = nib.as_closest_canonical(img)
    t1_numpy = normalize(t1_numpy)
    t2_numpy = normalize(t2_numpy)
    t1ce_numpy = normalize(t1ce_numpy)
    flair_numpy = normalize(flair_numpy)
    seg_numpy = normalize(seg_numpy, seg=True)
    
    return t1_numpy, t2_numpy, t1ce_numpy, flair_numpy, seg_numpy
    #return [np.max(array) - np.min(array) for array in np.nonzero(t1_numpy)]
    
def preprocessingMRbrain(data_path, extension):
    image_path = data_path[0]
    image_name = data_path[1]
    img_t1  = nib.load(os.path.join(image_path, f'pre/reg_T1.{extension}'))
    img_ir  = nib.load(os.path.join(image_path, f'pre/reg_IR.{extension}'))
    img_flair  = nib.load(os.path.join(image_path, f'pre/FLAIR.{extension}'))
    img_seg  = nib.load(os.path.join(image_path, f'segm.{extension}'))
    
    t1_numpy = img_t1.get_fdata()
    ir_numpy = img_ir.get_fdata()
    flair_numpy = img_flair.get_fdata()
    seg_numpy = img_seg.get_fdata()
    print(np.max(seg_numpy))
    #canonical_img = nib.as_closest_canonical(img)
    t1_numpy = normalize(t1_numpy)
    ir_numpy = normalize(ir_numpy)
    flair_numpy = normalize(flair_numpy)
 
    return t1_numpy, ir_numpy, flair_numpy, seg_numpy

def normalize(img_nii, seg=False):
    img = img_nii.get_fdata()
    if seg:
        img = np.array(img, dtype = np.int32)
        # Make the label set from [0,1,2,4] to [0,1,2,3]
        img[img==4] = 3
    else:
        img = np.array(img, dtype = np.float32)
        mask = img != 0
        img[mask] = (img[mask] - np.mean(img[mask])) / np.std(img[mask])

    return img

def seg_range(seg):
    seg_index = np.nonzero(seg[:, :, :])
    return [seg.shape[i] // 2 if len(seg_index[i]) == 0 else int(np.mean(seg_index[i])) for i in range(3)]

#Preprocessing
#Detail in https://torchio.readthedocs.io/transforms/preprocessing.html
def rescale_intensity(img, min=0, max=1):
    transform = tio.RescaleIntensity(
        out_min_max = (min,max),
        percentiles = (0.2, 99.8),
    )
    return transform(img)

def crop(img, w, h, d):
    transform = tio.Crop(cropping = (w,h,d)) #Center Crop
    return transform(img)

def pad(img, w, h, d):
    transform = tio.Pad(padding = (w,h,d), padding_mode = 'reflect')
    return transform(img)

#Augmentation
#Detail in https://torchio.readthedocs.io/transforms/augmentation.html
def random_elastic_deformation(img, ctrl_points=(6,6,6), locked_borders = 2):
    transform = RandomElasticDeformation(
        num_control_points = ctrl_points, # Number of control points along each dimension of the coarse grid
        locked_borders = locked_borders, # If 0, all displacement vectors are kept.
        image_interpolation = 'nearest',
        label_interpolation = 'nearest',
    )
    deform_img = transform(img)
    return deform_img[0, :, :, :], deform_img[1, :, :, :], deform_img[2, :, :, :], deform_img[3, :, :, :], deform_img[4, :, :, :]

def random_scale(img, x, y, z):
    transform = tio.RandomAffine(scales = (x,y,z)) #scale (1-x,1+x), (1-y,1+y), (1-z,1+z) for each axis
    return transform(img)

def random_rotation(img, theta1, theta2, theta3):
    transform = tio.RandomAffine(degrees = (theta1,theta2,theta3)) #rotate (-theta1,+theta1), (-theta2,+theta2), (-theta3,+theta3) for each axis
    return transform(img)

def random_translation(img, x, y, z):
    transform = tio.RandomAffine(translation = (x,y,z)) #translate (-x,+x), (-y,+y), (-z,+z) for each axis in mm unit
    return transform(img)

def random_affine(img, scale, rot, trans, label = 0): #Scale+Rotation+Translation at once
    mode = 'nearest' if label else 'linear'
    transform = tio.RandomAffine(
        scales = (scale[0], scale[0], scale[1], scale[1], scale[2], scale[2]), #scale (scale1, scale2) for each axis
        degrees = (rot[0], rot[0], rot[1], rot[1], rot[2], rot[2]), #rotate (rot1, rot2) for each axis
        translation = (trans[0], trans[0], trans[1], trans[1], trans[2], trans[2]), #translate (trans1, trans2) for each axis in mm unit
        image_interpolation = mode,
    )
    return transform(img)

def random_blur(img, std = (0.5,1.5)):
    transform = tio.RandomBlur(std = std) #If two values (a,b) are provided, then Gaussian kernels used to blur the image along each axis, where sigma ~ U(a,b) 
    img = transform(img)
    return img

def random_noise(img, mean = 0, std = (0, 0.1)):
    transform = tio.RandomNoise(mean = mean, std = std) # Mean of the Gaussian distribution from which the noise is sampled. Standard Deviation of the Gaussian distribution from which the noise is sampled.
    img = transform(img)
    return img

def random_gamma(img, gamma = (-0.35, 0.4)): #Randomly change contrast of an image by raising its values to the power gamma
    transform = tio.RandomGamma(log_gamma = gamma) #Tuple (a,b) to compute the exponent gamma = e^(beta), where beta ~ U(a,b)
    img = transform(img)
    return img

def random_flip(img, axes=(0,1,2), p=1.0):
    transform = tio.RandomFlip(axes = axes, flip_probability = p) #Index or tuple of indices of the spatial dimensions along which the image might be flipped. If they are integers, they must be in (0, 1, 2).
    return transform(img)

def random_jitter(img):
    img += (np.random.rand() - 0.5) * 2 / 10 # -0.1~0.1
    return img

def random_brightness(img):
    img = img * np.random.uniform(0.7,1.3)
    return img

def random_contrast(img):
    max_intensity = np.max(img)
    min_intensity = np.min(img)
    img = img * np.random.uniform(0.65,1.5)
    img = np.clip(img, min_intensity, max_intensity)
    return img

def preprocessing(data_path, extension):
    image_path = data_path[0]
    image_name = data_path[1]
    if 'hdf5' in image_path:
        f = h5py.File(image_path, 'r')
        t1_numpy = np.array(f.get('t1'), dtype = np.float32)
        t2_numpy = np.array(f.get('t2'), dtype = np.float32)
        t1ce_numpy = np.array(f.get('t1ce'), dtype = np.float32)
        flair_numpy = np.array(f.get('flair'), dtype = np.float32)
        seg_numpy = np.array(f.get('label'), dtype = np.int32)
    else:
        img_t1  = nib.load(os.path.join(image_path, image_name+'_t1.'+extension))
        img_t2  = nib.load(os.path.join(image_path, image_name+'_t2.'+extension))
        img_t1ce  = nib.load(os.path.join(image_path, image_name+'_t1ce.'+extension))
        img_flair  = nib.load(os.path.join(image_path, image_name+'_flair.'+extension))
        img_seg  = nib.load(os.path.join(image_path, image_name+'_seg.'+extension))
        #canonical_img = nib.as_closest_canonical(img)
        t1_numpy = normalize(img_t1)
        t2_numpy = normalize(img_t2)
        t1ce_numpy = normalize(img_t1ce)
        flair_numpy = normalize(img_flair)
        seg_numpy = normalize(img_seg, seg=True)
    return t1_numpy, t2_numpy, t1ce_numpy, flair_numpy, seg_numpy, seg_range(seg_numpy)

def preprocessingMRBrain(data_path, extension):
    image_path = data_path[0]
    image_name = data_path[1]
    if 'hdf5' in image_path:
        f = h5py.File(image_path, 'r')
        t1_numpy = np.array(f.get('t1'), dtype = np.float32)
        ir_numpy = np.array(f.get('ir'), dtype = np.float32)
        flair_numpy = np.array(f.get('flair'), dtype = np.float32)
        seg_numpy = np.array(f.get('label'), dtype = np.int32)
    return t1_numpy, ir_numpy, flair_numpy, seg_numpy, seg_range(seg_numpy)

def zero_padding(img, target):
    pad = [(t - i)//2 if (t - i)//2 >0 else 0 for i, t in zip(img.shape, target)]
    
    img_padded = np.zeros(target)
    img_padded[pad[0]:pad[0]+img.shape[0], pad[1]:pad[1]+img.shape[1], pad[2]:pad[2]+img.shape[2]] = img
    return img_padded

def resizing(img_nii, idx, target):
    img = img_nii.get_fdata()
    pad = [t - i if (t - i)//2 >0 else 0 for i, t in zip(img.shape, target)]
    img_padded = np.zeros([i+p for i, p in zip(img.shape, pad)])
    img_padded[pad[0]//2:pad[0]//2+img.shape[0], pad[1]//2:pad[1]//2+img.shape[1], pad[2]//2:pad[2]//2+img.shape[2]] = img
    idx = [i+p//2 for i,p in zip(idx,pad)]
    idx = [s - p // 2 if c + p // 2 > s else c for c, p, s in zip(idx, target, img_padded.shape)]
    idx = [p // 2 if c - p // 2 < 0 else c for c, p in zip(idx, target)] 
    crop_idx = [c - p // 2 for c, p in zip(idx, target)]
    img_resized = img_padded[crop_idx[0]:crop_idx[0]+target[0], crop_idx[1]:crop_idx[1]+target[1], crop_idx[2]:crop_idx[2]+target[2]]
    
    return img_resized

def data_augmentaion(t1_numpy, t2_numpy, t1ce_numpy, flair_numpy, seg_numpy):
    a, b, c = t1_numpy.shape[0], t1_numpy.shape[1], t1_numpy.shape[2]
    t1_numpy, t2_numpy, t1ce_numpy, flair_numpy, seg_numpy = t1_numpy.reshape(1, a, b, c), t2_numpy.reshape(1, a, b, c), t1ce_numpy.reshape(1, a, b, c), flair_numpy.reshape(1, a, b, c), seg_numpy.reshape(1, a, b, c)

    ##Elastic Deformation, probability of 0.3
    '''
    if np.random.rand() >= 0.7:
        concat = np.concatenate((t1_numpy, t2_numpy, t1ce_numpy, flair_numpy, seg_numpy), axis=0)
        t1_numpy[0,:,:,:], t2_numpy[0,:,:,:], t1ce_numpy[0,:,:,:], flair_numpy[0,:,:,:], seg_numpy[0,:,:,:] = random_elastic_deformation(concat)
    '''
    ##Augmentation(scale, rotation, translation), probability of 0.2
    rot = [0,0,0]
    if np.random.rand() <= 0.2:
        rot = []
        for i in range (3):
            rot.append(np.random.uniform(-30, 30))

    scale = [1, 1, 1]
    if np.random.rand() <= 0.2:
        scale = []
        for i in range (3):
            scale.append(np.random.uniform(0.7, 1.4))
    trans = [0,0,0]
    t1_numpy = random_affine(t1_numpy, scale, rot, trans, 0)
    t2_numpy = random_affine(t2_numpy, scale, rot, trans, 0)
    t1ce_numpy = random_affine(t1ce_numpy, scale, rot, trans, 0)
    flair_numpy = random_affine(flair_numpy, scale, rot, trans, 0)
    seg_numpy = random_affine(seg_numpy, scale, rot, trans, 1)
    
    #Gaussian noise
    if np.random.rand() <= 0.15:
        t1_numpy = random_noise(t1_numpy)
        t2_numpy = random_noise(t2_numpy)
        t1ce_numpy = random_noise(t1ce_numpy)
        flair_numpy = random_noise(flair_numpy)
    
    #Gaussian blur
    if np.random.rand() <= 0.2:
        if np.random.rand() <= 0.5:
            t1_numpy = random_blur(t1_numpy)
        if np.random.rand() <= 0.5:
            t2_numpy = random_blur(t2_numpy)
        if np.random.rand() <= 0.5:
            t1ce_numpy = random_blur(t1ce_numpy)
        if np.random.rand() <= 0.5:
            flair_numpy = random_blur(flair_numpy)

    #Brightness
    if np.random.rand() <= 0.15:
        t1_numpy = random_brightness(t1_numpy)
        t2_numpy = random_brightness(t2_numpy)
        t1ce_numpy = random_brightness(t1ce_numpy)
        flair_numpy = random_brightness(flair_numpy)
    
    #Contrast
    if np.random.rand() <= 0.15:
        t1_numpy = random_contrast(t1_numpy)
        t2_numpy = random_contrast(t2_numpy)
        t1ce_numpy = random_contrast(t1ce_numpy)
        flair_numpy = random_contrast(flair_numpy)
    '''
    ##Random Jitter, probability of 0.2
    if np.random.rand() >= 0.8:
        t1_numpy = random_jitter(t1_numpy)
        t2_numpy = random_jitter(t2_numpy)
        t1ce_numpy = random_jitter(t1ce_numpy)
        flair_numpy = random_jitter(flair_numpy)
    '''
    ##Gamma augmentation, probability of 0.3
    if np.random.rand() <= 0.15:
        t1_numpy = random_gamma(t1_numpy)
        t2_numpy = random_gamma(t2_numpy)
        t1ce_numpy = random_gamma(t1ce_numpy)
        flair_numpy = random_gamma(flair_numpy)

    #Flip
    if np.random.rand() <= 0.5:
        t1_numpy = random_flip(t1_numpy, axes=0)
        t2_numpy = random_flip(t2_numpy, axes=0)
        t1ce_numpy = random_flip(t1ce_numpy, axes=0)
        flair_numpy = random_flip(flair_numpy, axes=0)
        seg_numpy = random_flip(seg_numpy, axes=0)
    if np.random.rand() <= 0.5:
        t1_numpy = random_flip(t1_numpy, axes=1)
        t2_numpy = random_flip(t2_numpy, axes=1)
        t1ce_numpy = random_flip(t1ce_numpy, axes=1)
        flair_numpy = random_flip(flair_numpy, axes=1)
        seg_numpy = random_flip(seg_numpy, axes=1)
    if np.random.rand() <= 0.5:
        t1_numpy = random_flip(t1_numpy, axes=2)
        t2_numpy = random_flip(t2_numpy, axes=2)
        t1ce_numpy = random_flip(t1ce_numpy, axes=2)
        flair_numpy = random_flip(flair_numpy, axes=2)
        seg_numpy = random_flip(seg_numpy, axes=2)

    return t1_numpy.reshape(a, b, c), t2_numpy.reshape(a, b, c), t1ce_numpy.reshape(a, b, c), flair_numpy.reshape(a, b, c), seg_numpy.reshape(a, b, c), seg_range(seg_numpy.reshape(a,b,c))
