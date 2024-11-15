from __future__ import print_function
import os
import argparse
import shutil
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.utils.prune as prune
import torchvision
from torchvision import datasets, transforms
import utils
from utils import *
from tqdm import tqdm
import pickle
from function import *
from utils_1 import *
import logging
import random
from model import *

import torch.backends.cudnn as cudnn



parser = argparse.ArgumentParser(description='Pytorch PatDNN training')
parser.add_argument('--model',      default='WRN16-4_Q',          help = 'select model : ResNet20_Q, WRN16-4_Q, ResNet18_Q')
parser.add_argument('--dir',        default='/Data',           help='dataset root')
parser.add_argument('--dataset',    default='cifar10',          help='select dataset')
parser.add_argument('--batchsize',  default=512, type=int,      help='set batch size')
parser.add_argument('--lr',         default=1e-1, type=float,   help='set learning rate') # 6e-5
parser.add_argument('--epoch',      default=120, type=int,      help='set epochs') # 60
parser.add_argument('--exp',        default='test', type=str,   help='test or not')
parser.add_argument('--l2',         default=False, action='store_true', help='apply l3 regularization')
parser.add_argument('--scratch',    default=False, action='store_true', help='start from pretrain/scratch')
parser.add_argument('--no-cuda',    default=False, action='store_true', help='disables CUDA training')
parser.add_argument('--GPU', type=int, default=3)
parser.add_argument('--ab', type=int, default=32)
parser.add_argument('--wb', type=int, default=4)
parser.add_argument('--seed', type=int, default=230911)
parser.add_argument('--workers', type=int, default=8)

args = parser.parse_args()
print(args)


torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
np.random.seed(args.seed)
cudnn.benchmark = False
cudnn.deterministic = True
random.seed(args.seed)

GPU_NUM = args.GPU # GPU
# device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
# torch.cuda.set_device(device) # change allocation of current GPU
# print ('Current cuda device ', torch.cuda.current_device()) # check

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  # Arrange GPU devices starting from 0
os.environ["CUDA_VISIBLE_DEVICES"]= str(GPU_NUM)  # Set the GPU 2 to use
device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
print ('Current cuda device ', device)


logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

name_ = str(args.model) + '_' + str(args.dataset)

directory_save = './log/%s/pretrained'%(args.seed)
if not os.path.isdir(directory_save):
    os.makedirs(directory_save)

file_name = directory_save + '/%s_ab%d_wb%d.log'%(name_, args.ab, args.wb)

file_handler = logging.FileHandler(file_name)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)




use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
##### Load Dataset ####
print('\nPreparing Dataset...')
train_loader, test_loader = data_loader(args.dir, args.dataset, args.batchsize, args.workers)
print('Dataset Loaded')


print('\nDefined Model is loaded...')
if args.model == 'ResNet20_Q' :
    model = ResNet20_Q(args.ab, args.wb, block=BasicBlock_Q, num_blocks=[3,3,3], num_classes=10).cuda()

elif args.model == 'WRN16-4_Q' :
    if args.dataset == 'cifar10':
        model = Wide_ResNet_Q(args.ab, args.wb, block=Wide_BasicBlock_Q, num_blocks=[2,2,2], scale=4, num_classes=10).cuda()
    else:
        model = Wide_ResNet_Q(args.ab, args.wb, block=Wide_BasicBlock_Q, num_blocks=[2,2,2], scale=4, num_classes=100).cuda()

elif args.model == 'ResNet18_Q':
    model = ResNet18_Q(BasicBlock, [2, 2, 2, 2], args.ab, args.wb, num_classes = 1000).cuda()


def eval(model, test_loader):
    correct = 0
    total = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    with torch.no_grad():
        for i, (img, target) in enumerate(test_loader):
            img = img.to(device)
            out = model(img)
            pred = out.max(1)[1].detach().cpu().numpy()
            target = target.cpu().numpy()
            correct += (pred==target).sum()
            total += len(target)
    return correct / total




def train_model(model, train_loader, test_loader):
    logger.info("="*100)
    best_acc = -1
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    if args.model == 'ResNet18_Q':
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60, 90], gamma=0.1)
        # T=30
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T, 1e-4) 
    elif args.model == 'WRN16-4_Q':
        T = 20
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T, 1e-4)
    else :
        T = 20
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T, 1e-4)


    print(f'TRAINING START!')
    for epoch in range(args.epoch):
        model.train()
        cnt = 0
        loss_sum = 0
        for i, (img, target) in enumerate(train_loader):
            cnt += 1
            img, target = img.to(device), target.to(device)
            optimizer.zero_grad()
            out = model(img)
            loss = F.cross_entropy(out, target)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
        
        scheduler.step()
        loss_sum = loss_sum / cnt
        model.eval()
        acc = eval(model, test_loader)
        print(f'Epochs : {epoch+1}, Accuracy : {acc}')
        logger.info("Epoch %d/%d, Acc=%.4f"%(epoch+1, args.epoch, acc))
        

        if acc > best_acc :
            best_acc = acc
            print('Best accuracy is updated! at epoch %d/%d: %.4f '%(epoch, args.epoch, best_acc))
            logger.info('Best accuracy is : %.4f '%(best_acc))
            torch.save(model.state_dict(), directory_save + '/%s_ab%d_wb%d.pt'%(name_, args.ab, args.wb))
    logger.info('Final best accuracy is : %.4f '%(best_acc))

    

def main():
    train_loader, test_loader = data_loader(args.dir, args.dataset, args.batchsize, args.workers)


    train_model(model, train_loader, test_loader)

    # input = torch.randn(1, 3, 32, 32).cuda()
    # flops, params = profile(model, inputs=(input,))


if __name__=='__main__':
  main()



