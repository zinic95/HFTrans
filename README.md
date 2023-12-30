# HFTrans
Hybrid-Fusion Transformer for Multisequence MRI: Advancing Medical Image Segmentation
## Requriements
Install python requirements:
```
pip install -r requirements.txt
```
## Data Accqusition
The multisequence MRI brain tumor dataset (BraTS2020) could be acquired from [here](https://www.med.upenn.edu/cbica/brats-2019/).

The multisequence MRI brain structure dataset (MRBrainS18) could be acquired from [here](https://mrbrains18.isi.uu.nl/).

## Data Preprocseeing
Data preprocessing is needed to convert the .nii(.nii.gz) files to .hdf files and data normalization.
```
python3 preprocessMRBrain.py
python3 preprocessBraTS.py
```

We suggest the following folder structure for the training in cross-validation.
```
data/
--- fold1/
--- fold2/
--- ...
--- fold(n-1)/
--- fold(n)/
```

## Training
For BraTS2020
```
  python3 train.py --dataset your_BraTS_folder --data BraTS --patch-size 128 128 128 --crossvalid --fold 1
```
For MRBrainS18
```
  python3 train.py --dataset your_MRB_folder --data MRBrainS --patch-size 128 128 32 --crossvalid --fold 1
```

## Evaluation
For BraTS2020
```
  python eval.py --dataset your_BraTS_folder --data BraTS --patch-size 128 128 128 --weight your_weight_path --crossvalid --fold 1 --tta
```
For MRBrainS18
```
  python eval.py --dataset your_MRB_folder --data MRBrainS --patch-size 128 128 32 --weight your_weight_path --crossvalid --fold 1 --tta
```

## Reference
  [ViT-V-Net](https://github.com/junyuchen245/ViT-V-Net_for_3D_Image_Registration_Pytorch)
