import logging
import os
import time
import random, math
import argparse

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
#import cc3d
from torch.utils.data import DataLoader
from torch.nn.modules.utils import _triple
from scipy.ndimage.filters import gaussian_filter

from tensorboardX import SummaryWriter

from network.networks import *
from data.dataset import *

logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s',
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S"
)


def get_gaussian(patch_size, sigma_scale=1. / 8):
    tmp = np.zeros(patch_size)
    center_coords = [i // 2 for i in patch_size]
    sigmas = [i * sigma_scale for i in patch_size]
    tmp[tuple(center_coords)] = 1
    gaussian_importance_map = gaussian_filter(tmp, sigmas, 0, mode='constant', cval=0)
    gaussian_importance_map = gaussian_importance_map / np.max(gaussian_importance_map) * 1
    gaussian_importance_map = gaussian_importance_map.astype(np.float32)

    # gaussian_importance_map cannot be 0, otherwise we may end up with nans!
    gaussian_importance_map[gaussian_importance_map == 0] = np.min(
        gaussian_importance_map[gaussian_importance_map != 0])

    return gaussian_importance_map


def eval(model, device, images, patch_size, tta, data, writer, iter):
    with torch.no_grad():
        model.eval()

        images = [img.to(device) for img in images]
        input_img = torch.cat((images[0:-1]), dim=1)
        label = images[-1]
        if 'BraTS' in data:
            pred = torch.zeros(input_img.shape[0], 4, input_img.shape[2], input_img.shape[3], input_img.shape[4]).to(device)
        elif data == 'MRBrainS':
            pred = torch.zeros(input_img.shape[0], 11, input_img.shape[2], input_img.shape[3], input_img.shape[4]).to(device)
        size = input_img.shape[2:5]
        stride = [p // 2 for p in patch_size]
        importance_map = torch.Tensor(get_gaussian(patch_size)).to(device)
        sm = nn.Softmax(dim=1)
        x = 0
        for i in range(1 + math.ceil((size[0] - patch_size[0]) / stride[0])):
            y = 0
            for j in range(1 + math.ceil((size[1] - patch_size[1]) / stride[1])):
                z = 0
                for k in range(1 + math.ceil((size[2] - patch_size[2]) / stride[2])):
                    input_patch = input_img[:,:,x: x + patch_size[0], y: y + patch_size[1], z: z + patch_size[2]]
                    # TTA
                    if tta:
                        flip_dims = []
                        img_patch = torch.zeros(8, input_patch.shape[1], input_patch.shape[2], input_patch.shape[3], input_patch.shape[4])
                        for n in range(8):
                            flip_dim = []
                            if n // 4 == 1:
                                flip_dim.append(2)
                            if (n % 4) // 2  == 1:
                                flip_dim.append(3)
                            if n % 4 % 2 == 1:
                                flip_dim.append(4)
                            flip_dims.append(flip_dim)
                            img_patch[n:n+1,:,:,:,:] = torch.flip(input_patch, flip_dim)
                        segs, *_ = model(img_patch)
                        for n in range(8):
                            seg = torch.flip(segs[n:n+1,:,:,:,:], flip_dims[n])
                            pred[:,:,x: x + patch_size[0], y: y + patch_size[1], z: z + patch_size[2]] += seg * importance_map
                    else:
                        seg, *_ = model(input_patch)
                        pred[:,:,x: x + patch_size[0], y: y + patch_size[1], z: z + patch_size[2]] += seg * importance_map
                    z += stride[2]
                    if z + patch_size[2] > size[2]:
                        z = size[2] - patch_size[2]
                y += stride[1]
                if y + patch_size[1] > size[1]:
                    y = size[1] - patch_size[1]
            x += stride[0]
            if x + patch_size[0] > size[0]:
                x = size[0] - patch_size[0]
        
        pred_sm = sm(pred)
        pred = torch.argmax(pred_sm, dim = 1)
        pred[input_img[:,0,:,:,:]==0] = 0

        dices = []
        hd95s = []
        vss = []
        if 'BraTS' in data:
            dices.append(diceCoeff(pred, label, [3]).item())
            dices.append(diceCoeff(pred, label, [1,3]).item())
            dices.append(diceCoeff(pred, label, [1,2,3]).item())
            hd95s.append(hausdorff95(pred, label, [3]))
            hd95s.append(hausdorff95(pred, label, [1,3]))
            hd95s.append(hausdorff95(pred, label, [1,2,3]))
            pred = torch.div(pred, 3)
            label = torch.div(label, 3)
           # tensorboard.
            writer.add_image('eval/pred_patch', torch.unsqueeze(pred, 1)[0,0:1,:,:,size[2]//2], iter)
            writer.add_image('eval/label_patch', torch.unsqueeze(label, 1)[0,0:1,:,:,size[2]//2], iter)

            writer.add_scalar('eval/dice_EN', dices[0] * 100, iter)
            writer.add_scalar('eval/dice_TC', dices[1] * 100, iter)
            writer.add_scalar('eval/dice_WT', dices[2] * 100, iter)
            writer.add_scalar('eval/HD95_EN', hd95s[0], iter)
            writer.add_scalar('eval/HD95_TC', hd95s[1], iter)
            writer.add_scalar('eval/HD95_WT', hd95s[2], iter)
        elif data == 'MRBrainS':
            for i in [1,2,3,4,5,6,7,8]:
                dices.append(diceCoeff(pred, label, [i]).item())
                hd95s.append(hausdorff95(pred, label, [i], voxelspacing=(0.958, 0.958, 3)))
                vss.append(volumeSimillarity(pred, label, [i]))

        return dices, hd95s, vss

if __name__ == "__main__":
    # Version of Pytorch
    logging.info("Pytorch Version:%s" % torch.__version__)

    parser = argparse.ArgumentParser(description='BraTS')

    parser.add_argument('--dataset', type=str, required=True,
                        help='path of processed dataset')
    parser.add_argument('--patch-size', '--list', nargs='+', required=True, metavar='N',
                        help='3D patch-size x y z')
    parser.add_argument('--extention', type=str, default='hdf5', metavar='N',
                        help='file extention format (default: hdf5)')
    parser.add_argument('--data', type=str, required=True, metavar='N',
                        help='name of the dataset')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 0)')
    parser.add_argument('--gpu', type=str, default='0', metavar='N',
                        help='Select the GPU (defualt 0)')
    parser.add_argument('--weight', type=str, default='./weights/best_valid.pth',
                        help='trained weights of model for eval')
    parser.add_argument('--crossvalid', action='store_true',
                        help='Training using crossfold')
    parser.add_argument('--fold', type=int, default=1, metavar='N',
                        help='Valid fold num (1~5)')
    parser.add_argument('--tta', action='store_true',
                        help='Test Time Augmentation')

    args = parser.parse_args()
    args.patch_size = list(map(int, args.patch_size))
    patch_size = args.patch_size
    batch_size = 1

    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Torch use the device: %s" % device)

    if args.data == 'BraTS':
        config_vit = CONFIGS['HFTrans5_16']
        model = HFTrans(config_vit, img_size=patch_size, input_channels=4, num_classes=4, num_encoders=5, vis=False).to(device)
        test_dataset = BraTSDataset(args.dataset, patch_size = patch_size, subset='test', extention= args.extention, crossvalid=args.crossvalid, valid_fold=args.fold)
    elif args.data == 'BraTSb2s':
        config_vit = CONFIGS['HFTrans_16b2s']
        model = HFTransb2s(config_vit, img_size=patch_size, input_channels=4, num_classes=4, num_encoders=5, vis=False).to(device)
        test_dataset = BraTSDataset(args.dataset, patch_size = patch_size, subset='test', extention= args.extention, crossvalid=args.crossvalid, valid_fold=args.fold)
    elif args.data == 'BraTSsim':
        config_vit = CONFIGS['HFTrans_16']
        model = HFTransSimple(config_vit, img_size=patch_size, input_channels=4, num_classes=4, num_encoders=5, vis=False).to(device)
        test_dataset = BraTSDataset(args.dataset, patch_size = patch_size, subset='test', extention= args.extention, crossvalid=args.crossvalid, valid_fold=args.fold)
    elif args.data == 'BraTSsc':
        config_vit = CONFIGS['HFTrans_16']
        model = HFTransSC(config_vit, img_size=patch_size, input_channels=4, num_classes=4, num_encoders=5, vis=False).to(device)
        test_dataset = BraTSDataset(args.dataset, patch_size = patch_size, subset='test', extention= args.extention, crossvalid=args.crossvalid, valid_fold=args.fold)
    elif args.data == 'BraTSca':
        config_vit = CONFIGS['HFTrans5_16']
        model = HFTransCA(config_vit, img_size=patch_size, input_channels=4, num_classes=4, num_encoders=5, vis=False).to(device)
        test_dataset = BraTSDataset(args.dataset, patch_size = patch_size, subset='test', extention= args.extention, crossvalid=args.crossvalid, valid_fold=args.fold)
    elif args.data == 'BraTSe':
        config_vit = CONFIGS['HFTrans_16']
        model = HFTrans(config_vit, img_size=patch_size, input_channels=4, num_classes=4, num_encoders=1, vis=False).to(device)
        test_dataset = BraTSDataset(args.dataset, patch_size = patch_size, subset='test', extention= args.extention, crossvalid=args.crossvalid, valid_fold=args.fold)
    elif args.data == 'BraTSm':
        config_vit = CONFIGS['HFTrans4_16']
        model = HFTrans_middle(config_vit, img_size=patch_size, input_channels=4, num_classes=4, num_encoders=4, vis=False).to(device)
        test_dataset = BraTSDataset(args.dataset, patch_size = patch_size, subset='test', extention= args.extention, crossvalid=args.crossvalid, valid_fold=args.fold)
    elif args.data == 'MRBrainS':
        config_vit = CONFIGS['HFTrans4_32']
        model = HFTrans(config_vit, img_size=patch_size, input_channels=3, num_classes=11, num_encoders=4, vis=False).to(device)
        test_dataset = MRBrainDataset(args.dataset, patch_size = patch_size, subset='test', extention= args.extention, crossvalid=args.crossvalid, valid_fold=args.fold)
    else:
        print(f'invalid dataset name {args.data}')
        exit(0)
    model = nn.DataParallel(model)

    # Load the weight 
    if('latest_checkpoints_' in args.weight):
        checkpoint = torch.load(args.weight)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(torch.load(args.weight))

    generator = torch.Generator()
    generator.manual_seed(args.seed)

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, generator=generator)
                
    writer = SummaryWriter('runs/eval/'+args.weight.split('/')[-1]+'/'+str(time.time()))
 
    if "BraTS" in args.data: 
        labels = ['ET', 'TC', 'WT']
        dices_cate = [[] for i in labels]
        hd95s_cate = [[] for i in labels]
    elif args.data == "MRBrainS":
        labels = ['GM', 'BG', 'WM', 'WML', 'CSF', 'Vent', 'Cereb', 'BS']
        dices_cate = [[] for i in labels]
        hd95s_cate = [[] for i in labels]
        vss_cate = [[] for i in labels]

    test_iterator = iter(test_loader)
    for i in range(len(test_loader)):
        images = next(test_iterator)
        dices, hd95s, vss = eval(model, device, images, patch_size, args.tta, args.data, writer, i)
        for j in range(len(dices)):
            dices_cate[j].append(dices[j])
            hd95s_cate[j].append(hd95s[j])
            if args.data == 'MRBrainS':
                vss_cate[j].append(vss[j])
                
    print('########################################')
    print(args.dataset + f'/fold{args.fold}')
    print(args.weight)
    for i in range(len(labels)):
        print(f'Average Dice Score of {labels[i]}: {np.mean(dices_cate[i]) * 100}')
    for i in range(len(labels)):
        print(f'Average Hausdorff Distacne95 of {labels[i]}: {np.mean(hd95s_cate[i])}')
    if args.data == 'MRBrainS':
        for i in range(len(labels)):
                print(f'Average Volumetric Similiarity of {labels[i]}: {np.mean(vss_cate[i])}')
    print('########################################')

