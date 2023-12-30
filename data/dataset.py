import os

from torch.utils.data import Dataset
import numpy as np
from utils.utils import *
from .data_preprocessing import *

class BraTSDataset(Dataset):
    """BraTS Dataset""" 
    def __init__(self,
                 dataset_path,
                 patch_size,
                 extention,
                 subset='train',
                 crossvalid= True,
                 valid_fold=1,
                 ):

        self.idx = 1
        self.patch_size = patch_size
        self.extention = extention
        self.hdf5 = 'hdf5' in dataset_path
        self.subset = subset
        self.crossvalid = crossvalid
        if self.crossvalid:
            if subset == 'train':
                self.data_path = []
                for i in [1,2,3,4,5]:
                    if i == valid_fold:
                        continue
                    self.dataset_path = os.path.join(dataset_path, 'fold'+ str(i))
                    img_names = [d for d in os.listdir(self.dataset_path) if d.startswith('BraTS')]
                    img_names.sort()
                    # N x 2
                    data_path = [[os.path.join(self.dataset_path, name), name] for name in img_names]
                    self.data_path.extend(data_path)
            elif subset == 'valid' or subset == 'test':
                self.dataset_path = os.path.join(dataset_path, 'fold'+ str(valid_fold))
                img_names = [d for d in os.listdir(self.dataset_path) if d.startswith('BraTS')]
                img_names.sort()
                # N x 2
                self.data_path = [[os.path.join(self.dataset_path, name), name] for name in img_names]
        else:
            self.dataset_path = os.path.join(dataset_path, subset)
            img_names = [d for d in os.listdir(self.dataset_path) if d.startswith('BraTS')]
            img_names.sort()
            # N x 2
            self.data_path = [[os.path.join(self.dataset_path, name), name] for name in img_names]
        
    def __len__(self):
        return len(self.data_path)

    def __getitem__(self, idx):
        
        t1_img, t2_img, t1ce_img, flair_img, label, center = preprocessing(self.data_path[idx], self.extention)

        if self.subset == 'test':
            return test_img(t1_img, t2_img, t1ce_img, flair_img, label)
        else:
            if self.subset == 'train':
                t1_img, t2_img, t1ce_img, flair_img, label, center = data_augmentaion(t1_img, t2_img, t1ce_img, flair_img, label)
            t1_patch, t2_patch, t1ce_patch, flair_patch, label_patch = extract_patch(t1_img, t2_img, t1ce_img, flair_img, label, center, self.patch_size)
            return t1_patch, t2_patch, t1ce_patch, flair_patch, label_patch

class MRBrainDataset(Dataset):
    """MRBrain18 Dataset""" 
    def __init__(self,
                 dataset_path,
                 patch_size,
                 extention,
                 subset='train',
                 crossvalid= True,
                 valid_fold=1,
                 ):

        self.idx = 1
        self.patch_size = patch_size
        self.extention = extention
        self.hdf5 = 'hdf5' in dataset_path
        self.subset = subset
        self.crossvalid = crossvalid
        if self.crossvalid:
            if subset == 'train':
                self.data_path = []
                for i in [1,2,3,4,5,6,7]:
                    if i == valid_fold:
                        continue
                    self.dataset_path = os.path.join(dataset_path, 'fold'+ str(i))
                    img_names = [d for d in os.listdir(self.dataset_path) if 'hdf5' in d]
                    img_names.sort()
                    # N x 2
                    data_path = [[os.path.join(self.dataset_path, name), name] for name in img_names]
                    self.data_path.extend(data_path)
            elif subset == 'valid' or subset == 'test':
                self.dataset_path = os.path.join(dataset_path, 'fold'+ str(valid_fold))
                img_names = [d for d in os.listdir(self.dataset_path) if 'hdf5' in d]
                img_names.sort()
                # N x 2
                self.data_path = [[os.path.join(self.dataset_path, name), name] for name in img_names]
        else:
            self.dataset_path = os.path.join(dataset_path, subset)
            img_names = [d for d in os.listdir(self.dataset_path) if 'hdf5' in d]
            img_names.sort()
            # N x 2
            self.data_path = [[os.path.join(self.dataset_path, name), name] for name in img_names]
        
    def __len__(self):
        return len(self.data_path)

    def __getitem__(self, idx):
        t1_img, ir_img, flair_img, label, center = preprocessingMRBrain(self.data_path[idx], self.extention)

        if self.subset == 'test':
            t1_patch, ir_patch, flair_patch, _, label_patch = test_img(t1_img, ir_img, flair_img, flair_img, label)
            return t1_patch, ir_patch, flair_patch, label_patch
        else:
            if self.subset == 'train':
                t1_img, ir_img, flair_img, _, label, center = data_augmentaion(t1_img, ir_img, flair_img, flair_img, label) 
            t1_patch, ir_patch, flair_patch, _, label_patch = extract_patch(t1_img, ir_img, flair_img, flair_img, label, center, self.patch_size)
            return t1_patch, ir_patch, flair_patch, label_patch
