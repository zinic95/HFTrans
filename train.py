import logging
import os
import time
import random
import argparse

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from scipy.ndimage import zoom

from tensorboardX import SummaryWriter

from network.networks import *
from data.dataset import *

logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s',
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S"
)

def set_seed(seed):
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def train(model, device, patches, optimizer, data):
    model.train()
    #optimizer.zero_grad()

    patches = [patch.to(device) for patch in patches]
    input_patch = torch.cat((patches[0:-1]), dim=1)
    label_patch = patches[-1]
    seg, *_ = model(input_patch)
    ce = nn.CrossEntropyLoss()
    ce_loss = ce(seg, label_patch)

    sm = nn.Softmax(dim=1)
    seg_sm = sm(seg.clone())
    if 'BraTS' in data:
        dice_EN = diceCoeff(seg_sm, label_patch, [3], loss=True)
        dice_TC = diceCoeff(seg_sm, label_patch, [1,3], loss=True)
        dice_WT = diceCoeff(seg_sm, label_patch, [1,2,3], loss=True)
        
        total_loss = 1.0 * ce_loss + ((1 - dice_EN) + (1 - dice_TC) + (1 - dice_WT))
    elif data == 'MRBrainS':
        dice_CGM = diceCoeff(seg_sm, label_patch, [1], loss=True)
        dice_BG = diceCoeff(seg_sm, label_patch, [2], loss=True)
        dice_WM = diceCoeff(seg_sm, label_patch, [3], loss=True)
        dice_WML = diceCoeff(seg_sm, label_patch, [4], loss=True)
        dice_CF = diceCoeff(seg_sm, label_patch, [5], loss=True)
        dice_V = diceCoeff(seg_sm, label_patch, [6], loss=True)
        dice_Ce = diceCoeff(seg_sm, label_patch, [7], loss=True)
        dice_BS = diceCoeff(seg_sm, label_patch, [8], loss=True)
        dice_In = diceCoeff(seg_sm, label_patch, [9], loss=True)
        dice_Ot = diceCoeff(seg_sm, label_patch, [10], loss=True)

        total_loss = 1.0 * ce_loss + ((1 - dice_CGM) + (1 - dice_BG) + (1 - dice_WM) + (1 - dice_WML) + (1 - dice_CF) + (1 - dice_V) + (1 - dice_Ce) + (1 - dice_BS) + (1 - dice_In) + (1 - dice_Ot))
    
    logging.info(f"Train total Loss: {total_loss.item():.6f}, Cross Entrophy Loss: {ce_loss.item():.6f}")
    if math.isnan(total_loss.item()):
        for param in model.parameters():
            print(param.data)
    # optimize the parameters
    total_loss.backward()
    #optimizer.step()

    return total_loss.item()

def valid(model, device, patches, data):
    with torch.no_grad():
        model.eval()

        patches = [patch.to(device) for patch in patches]
        input_patch = torch.cat((patches[0:-1]), dim=1)
        label_patch = patches[-1]

        seg, *_ = model(input_patch)

        sm = nn.Softmax(dim=1)
        seg_sm = sm(seg.clone())
        dices = []

        if 'BraTS' in data:
            dice_EN = diceCoeff(seg_sm, label_patch, [3], loss=True)
            dice_TC = diceCoeff(seg_sm, label_patch, [1,3], loss=True)
            dice_WT = diceCoeff(seg_sm, label_patch, [1,2,3], loss=True)
            dices.append(dice_EN.item())
            dices.append(dice_TC.item())
            dices.append(dice_WT.item())
            total_loss = 1/3 *((1 - dice_EN) + (1 - dice_TC) + (1 - dice_WT))
        elif data == 'MRBrainS':
            dice_CGM = diceCoeff(seg_sm, label_patch, [1], loss=True)
            dice_BG = diceCoeff(seg_sm, label_patch, [2], loss=True)
            dice_WM = diceCoeff(seg_sm, label_patch, [3], loss=True)
            dice_WML = diceCoeff(seg_sm, label_patch, [4], loss=True)
            dice_CF = diceCoeff(seg_sm, label_patch, [5], loss=True)
            dice_V = diceCoeff(seg_sm, label_patch, [6], loss=True)
            dice_Ce = diceCoeff(seg_sm, label_patch, [7], loss=True)
            dice_BS = diceCoeff(seg_sm, label_patch, [8], loss=True)
            dice_In = diceCoeff(seg_sm, label_patch, [9], loss=True)
            dice_Ot = diceCoeff(seg_sm, label_patch, [10], loss=True)
            dices.append(dice_CGM.item())
            dices.append(dice_BG.item())
            dices.append(dice_WM.item())
            dices.append(dice_WML.item())
            dices.append(dice_CF.item())
            dices.append(dice_V.item())
            dices.append(dice_Ce.item())
            dices.append(dice_BS.item())
            dices.append(dice_In.item())
            dices.append(dice_Ot.item())
            total_loss = 1/10 *((1 - dice_CGM) + (1 - dice_BG) + (1 - dice_WM) + (1 - dice_WML) + (1 - dice_CF) + (1 - dice_V) + (1 - dice_Ce) + (1 - dice_BS) + (1 - dice_In) + (1 - dice_Ot))
        
        logging.info(f"Valid Total Loss: {total_loss.item():.6f}") 
        return total_loss.item(), dices

if __name__ == "__main__":
    # Version of Pytorch
    logging.info("Pytorch Version:%s" % torch.__version__)

    parser = argparse.ArgumentParser(description='BraTS')

    parser.add_argument('--dataset', type=str, required=True,
                        help='path of processed dataset')
    parser.add_argument('--identifier', type=str, required=True, metavar='N',
                        help='Select the identifier for file name')
    parser.add_argument('--batch-size', type=int,  default=1, metavar='N',
                        help='input batch size for training (default: 1)')
    parser.add_argument('--acc-batch-multiplier', type=int,  default=1, metavar='N',
                        help='accumulated batch size for training (default: 1)')
    parser.add_argument('--patch-size', '--list', nargs='+', required=True, metavar='N',
                        help='3D patch-size x y z')
    parser.add_argument('--epoches', type=int, default=100, metavar='N',
                        help='number of epoches to train (default: 100)')
    parser.add_argument('--extention', type=str, default='hdf5', metavar='N',
                        help='file extention format (default: hdf5)')
    parser.add_argument('--data', type=str, required=True, metavar='N',
                        help='name of the dataset')
    parser.add_argument('--weights', type=str, default='./weights',
                        help='path of training weight')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints',
                        help='path of training snapshot')
    parser.add_argument('--seed', type=int, default=0, metavar='S',
                        help='random seed (default: 0)')
    parser.add_argument('--gpu', type=str, default='0', metavar='N',
                        help='Select the GPU (defualt 0)')
    parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                        help='number of epoches to log (default: 1)')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--crossvalid', action='store_true',
                        help='Training using crossfold')
    parser.add_argument('--fold', type=int, default=1, metavar='N',
                        help='Valid fold num')
    parser.add_argument('--resume', action='store_true',
                        help='resume training by loading last snapshot')
    args = parser.parse_args()
    args.patch_size = list(map(int, args.patch_size))
    acc = args.acc_batch_multiplier
    batch_size = args.batch_size
    patch_size = args.patch_size
    if not os.path.exists(args.weights):
        os.makedirs(args.weights)
    if not os.path.exists(args.checkpoints):
        os.makedirs(args.checkpoints)
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    torch.autograd.set_detect_anomaly(True)

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    logging.info("Torch use the device: %s" % device)

    if args.data == 'BraTS':
        config_vit = CONFIGS['HFTrans5_16']
        model = HFTrans(config_vit, img_size=patch_size, input_channels=4, num_classes=4, num_encoders=5, vis=False).to(device)
        train_dataset = BraTSDataset(args.dataset, patch_size = patch_size, subset='train', extention= args.extention, crossvalid=args.crossvalid, valid_fold=args.fold)
        valid_dataset = BraTSDataset(args.dataset, patch_size = patch_size, subset='valid', extention= args.extention, crossvalid=args.crossvalid, valid_fold=args.fold)
    elif args.data == 'BraTSb2s':
        config_vit = CONFIGS['HFTrans_16b2s']
        model = HFTransb2s(config_vit, img_size=patch_size, input_channels=4, num_classes=4, num_encoders=5, vis=False).to(device)
        train_dataset = BraTSDataset(args.dataset, patch_size = patch_size, subset='train', extention= args.extention, crossvalid=args.crossvalid, valid_fold=args.fold)
        valid_dataset = BraTSDataset(args.dataset, patch_size = patch_size, subset='valid', extention= args.extention, crossvalid=args.crossvalid, valid_fold=args.fold)
    elif args.data == 'BraTSsim':
        config_vit = CONFIGS['HFTrans_16']
        model = HFTransSimple(config_vit, img_size=patch_size, input_channels=4, num_classes=4, num_encoders=5, vis=False).to(device)
        train_dataset = BraTSDataset(args.dataset, patch_size = patch_size, subset='train', extention= args.extention, crossvalid=args.crossvalid, valid_fold=args.fold)
        valid_dataset = BraTSDataset(args.dataset, patch_size = patch_size, subset='valid', extention= args.extention, crossvalid=args.crossvalid, valid_fold=args.fold)
    elif args.data == 'BraTSca':
        config_vit = CONFIGS['HFTrans5_16']
        model = HFTransCA(config_vit, img_size=patch_size, input_channels=4, num_classes=4, num_encoders=5, vis=False).to(device)
        train_dataset = BraTSDataset(args.dataset, patch_size = patch_size, subset='train', extention= args.extention, crossvalid=args.crossvalid, valid_fold=args.fold)
        valid_dataset = BraTSDataset(args.dataset, patch_size = patch_size, subset='valid', extention= args.extention, crossvalid=args.crossvalid, valid_fold=args.fold)
    elif args.data == 'BraTSsc':
        config_vit = CONFIGS['HFTrans_16']
        model = HFTransSC(config_vit, img_size=patch_size, input_channels=4, num_classes=4, num_encoders=5, vis=False).to(device)
        train_dataset = BraTSDataset(args.dataset, patch_size = patch_size, subset='train', extention= args.extention, crossvalid=args.crossvalid, valid_fold=args.fold)
        valid_dataset = BraTSDataset(args.dataset, patch_size = patch_size, subset='valid', extention= args.extention, crossvalid=args.crossvalid, valid_fold=args.fold)
    elif args.data == 'BraTSe':
        config_vit = CONFIGS['HFTrans_16']
        model = HFTrans(config_vit, img_size=patch_size, input_channels=4, num_classes=4, num_encoders=1, vis=False).to(device)
        train_dataset = BraTSDataset(args.dataset, patch_size = patch_size, subset='train', extention= args.extention, crossvalid=args.crossvalid, valid_fold=args.fold)
        valid_dataset = BraTSDataset(args.dataset, patch_size = patch_size, subset='valid', extention= args.extention, crossvalid=args.crossvalid, valid_fold=args.fold)
    elif args.data == 'BraTSm':
        config_vit = CONFIGS['HFTrans4_16']
        model = HFTrans_middle(config_vit, img_size=patch_size, input_channels=4, num_classes=4, num_encoders=4, vis=False).to(device)
        train_dataset = BraTSDataset(args.dataset, patch_size = patch_size, subset='train', extention= args.extention, crossvalid=args.crossvalid, valid_fold=args.fold)
        valid_dataset = BraTSDataset(args.dataset, patch_size = patch_size, subset='valid', extention= args.extention, crossvalid=args.crossvalid, valid_fold=args.fold)
    elif args.data == 'MRBrainS':
        config_vit = CONFIGS['HFTrans4_32']
        model = HFTrans(config_vit, img_size=patch_size, input_channels=3, num_classes=11, num_encoders=4, vis=False).to(device)
        train_dataset = MRBrainDataset(args.dataset, patch_size = patch_size, subset='train', extention= args.extention, crossvalid=args.crossvalid, valid_fold=args.fold)
        valid_dataset = MRBrainDataset(args.dataset, patch_size = patch_size, subset='valid', extention= args.extention, crossvalid=args.crossvalid, valid_fold=args.fold)
    else:
        print(f'invalid dataset name {args.data}')
        exit(0)
    model = nn.DataParallel(model)


    generator = torch.Generator()
    generator.manual_seed(args.seed)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True, generator=generator)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True, generator=generator)
    train_iterator = iter(train_loader)

    total_iteration = args.epoches * len(train_loader)
    train_interval = args.log_interval * len(train_loader) 

    logging.info(f"total iter: {total_iteration}")

    # optimizer

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.99, nesterov=True)
    #optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    optimizer.zero_grad()
    
    iteration = 1
    best_train_loss, best_valid_loss = float('inf'), float('inf')

    if args.resume:
        logging.info("Resume Training: Load states from latest checkpoint.")
        checkpoint = torch.load(os.path.join(args.checkpoints, f'latest_checkpoint_{args.data}_{args.identifier}_fold{args.fold}.pth'))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        iteration = checkpoint['iteration']
        best_train_loss = checkpoint['best_train_loss']
        best_valid_loss = checkpoint['best_valid_loss'] 
                
    # Start Training 
    writer = SummaryWriter('runs/'+args.data+args.identifier+'_fold'+str(args.fold)+'/'+str(time.time()))
    
    epoch_train_loss = []
    epoch_valid_loss = []
    if "BraTS" in args.data: 
        labels = ['ET', 'TC', 'WT']
        epoch_valid_dices = [[] for i in labels]
    elif args.data == "MRBrainS":
        labels = ['GM', 'BG', 'WM', 'WML', 'CSF', 'Vent', 'Cereb', 'BS', 'In', 'Ot']
        epoch_valid_dices = [[] for i in labels]
    set_seed(args.seed)
    start_time = time.time()
    while iteration <= total_iteration:
        try:
            patches = next(train_iterator)

        except StopIteration:
            # restart the generator if the previous generator is exhausted.
            train_iterator = iter(train_loader)
            patches = next(train_iterator)
            # polynomial lr schedule
            optimizer.param_groups[0]['lr'] = args.lr * (1 - iteration / total_iteration)**0.9
        t_segloss = train(model, device, patches, optimizer, args.data)
        if iteration % acc == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            optimizer.zero_grad()
        epoch_train_loss.append(t_segloss)

        if (iteration % train_interval == 0):
            avg_train_loss = sum(epoch_train_loss) / len(epoch_train_loss)

            logging.info(f'Iter {iteration / train_interval}-{total_iteration / train_interval}: \t Loss: {avg_train_loss:.6f}\t')

            if avg_train_loss < best_train_loss:
                best_train_loss = avg_train_loss
                logging.info(f'--- Saving model at Avg Train Loss:{avg_train_loss:.6f}  ---')
                torch.save(model.state_dict(), os.path.join(args.weights, f'best_train_{args.data}_fold{args.fold}.pth'))

            # validation process
            valid_iterator = iter(valid_loader)
            for i in range(len(valid_loader)):
                patches = next(valid_iterator)
                v_segloss, v_dices = valid(model, device, patches, args.data)

                epoch_valid_loss.append(v_segloss)
                for j in range(len(v_dices)):
                    epoch_valid_dices[j].append(v_dices[j])

            avg_valid_loss = sum(epoch_valid_loss) / (len(epoch_valid_loss) + 1e-6)
            avg_valid_dices = []
            for i in range(len(labels)):
                avg_valid_dices.append(sum(epoch_valid_dices[i]) / (len(epoch_valid_dices[i]) + 1e-6))

            logging.info(f'Iter {iteration / train_interval}-{total_iteration / train_interval} eval: \t Loss: {avg_valid_loss:.6f}')
            if avg_valid_loss < best_valid_loss:
                best_valid_loss = avg_valid_loss
                logging.info(f'--- Saving model at Avg Valid Loss:{avg_valid_loss:.6f}  ---')
                torch.save(model.state_dict(), os.path.join(args.weights, f'best_valid_{args.data}_{args.identifier}_fold{args.fold}.pth'))
            
            writer.add_scalar('valid/total_loss', avg_valid_loss, iteration)
            for i in range(len(labels)):
                writer.add_scalar(f'valid/dice_{labels[i]}', avg_valid_dices[i] * 100, iteration)

            # save snapshot for resume training
            logging.info('--- Saving snapshot ---')
            torch.save({
                'iteration': iteration+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_train_loss': best_train_loss,
                'best_valid_loss': best_valid_loss,
            },
                os.path.join(args.checkpoints, f'latest_checkpoint_{args.data}_{args.identifier}_fold{args.fold}.pth'))
            

            logging.info(f"--- {time.time() - start_time} seconds ---")

            epoch_train_loss = []
            epoch_valid_loss = []
            epoch_valid_dices = [[] for i in labels]

            start_time = time.time()
        iteration += 1