from __future__ import print_function
import os
import argparse
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
# from tqdm import tqdm
from utils_1 import *
import logging
import random
from model import *

import torch.backends.cudnn as cudnn


##### Settings #########################################################################                      3x3에서는 patdnn을 따라가도록?
parser = argparse.ArgumentParser(description='Pytorch PatDNN training')
parser.add_argument('--dir',        default='/Data',           help='dataset root')
parser.add_argument('--model',      default='WRN16-4_Q',          help = 'select model : ResNet20_Q, WRN16-4_Q')
parser.add_argument('--dataset',    default='cifar100',          help='select dataset : cifar10, cifar100')
parser.add_argument('--batchsize',  default=512, type=int,      help='set batch size')
parser.add_argument('--lr',         default=0.1, type=float,   help='set learning rate') # 6e-5
parser.add_argument('--epoch',      default=200, type=int,      help='set epochs') # 60
parser.add_argument('--no-cuda',    default=False, action='store_true', help='disables CUDA training')
parser.add_argument('--GPU', type=int, default=1) 
parser.add_argument('--ab', type=int, default=32)
parser.add_argument('--wb', type=int, default=32)
parser.add_argument('--seed', type=int, default = 1992)
args = parser.parse_args()
print(args)

GPU_NUM = args.GPU # GPU
# device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
# torch.cuda.set_device(device) # change allocation of current GPU
# print ('Current cuda device ', torch.cuda.current_device()) # check

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  # Arrange GPU devices starting from 0
os.environ["CUDA_VISIBLE_DEVICES"]= str(GPU_NUM)  # Set the GPU 2 to use
device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
print ('Current cuda device ', device)

args.workers = 2

use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
#########################################################################################################
# ResNet code modified from original of [https://github.com/akamaster/pytorch_resnet_cifar10/blob/master/resnet.py]
# Modified version for our experiment.

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
np.random.seed(args.seed)
cudnn.benchmark = False
cudnn.deterministic = True
random.seed(args.seed)

directory_save = './pt_save/%s/%s'%(args.model, args.dataset)
if not os.path.isdir(directory_save):
    os.makedirs(directory_save)

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
    # Clear acc_list

    best_acc = -1

    T = 25
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay= 5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T, 1e-4)

    save_dic = directory_save + '/trained.pt'

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
        
        loss_sum = loss_sum / cnt
        model.eval()
        scheduler.step()
        acc = eval(model, test_loader)

        print(f'Epochs : {epoch+1}, Accuracy : {acc}')
        

        if acc > best_acc :
            best_acc = acc
            print('Best accuracy is updated! at epoch %d/%d: %.4f '%(epoch+1, args.epoch, best_acc))

            torch.save(model.state_dict(), save_dic)
    

def main():
    train_loader, test_loader = data_loader(args.dir, args.dataset, args.batchsize, args.workers)

    if args.model == 'ResNet20_Q' :
        if args.dataset == 'cifar10' :
            model = ResNet20_Q(args.ab, args.wb, block=BasicBlock_Q, num_blocks=[3,3,3], num_classes=10).cuda()
        elif args.dataset == 'cifar100' :
            model = ResNet20_Q(args.ab, args.wb, block=BasicBlock_Q, num_blocks=[3,3,3], num_classes=100).cuda()

    elif args.model == 'WRN16-4_Q' :
        if args.dataset == 'cifar10' :
            model = Wide_ResNet_Q(args.ab, args.wb, block=Wide_BasicBlock_Q, num_blocks=[2,2,2], scale=4, num_classes=10).cuda()
        elif args.dataset == 'cifar100' :
            model = Wide_ResNet_Q(args.ab, args.wb, block=Wide_BasicBlock_Q, num_blocks=[2,2,2], scale=4, num_classes=100).cuda()


    train_model(model, train_loader, test_loader)

if __name__=='__main__':
  main()





