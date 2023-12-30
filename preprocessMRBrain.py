import numpy as np
import nibabel as nib
import h5py
import os
from data.data_preprocessing import *

def main():
    SourceRootPath = f'MRBrain18'
    targetPath = f'hdf5/MRBrain18'
    AllFileList = os.listdir(SourceRootPath)

    if ('.ipynb_checkpoints' in AllFileList):
        AllFileList.pop(AllFileList.index('.ipynb_checkpoints'))
    AllFileList.sort()
    #print(AllFileList)

    for file in AllFileList:
        t1_numpy, ir_numpy, flair_numpy, seg_numpy = preprocessingMRbrain([os.path.join(SourceRootPath,file), file], "nii.gz")
        
        hdf5_path = os.path.join(targetPath, file + '.hdf5')
        f = h5py.File(hdf5_path, 'w')
        f.create_dataset('t1', data=t1_numpy)
        f.create_dataset('ir', data=ir_numpy)
        f.create_dataset('flair', data=flair_numpy)
        f.create_dataset('label', data=seg_numpy)
        f.close()

if __name__ == "__main__":
    main()