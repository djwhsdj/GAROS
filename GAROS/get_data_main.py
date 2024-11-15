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
# from model import *
import json
import torch.backends.cudnn as cudnn
import time
import datetime # datetime 라이브러리 import

from numpy import dot
from numpy.linalg import norm

from typing import Literal


from geneticalgorithm2 import geneticalgorithm2 as ga





def GA_selction_v1(model, args):
    idx = -1
    idxx = 0
    GA_list = []

    for name, param in model.named_parameters():
        if name.split('.')[-1] == "weight" and len(param.shape) == 4 and 'downsample' not in name and 'conv' in name:
            idx += 1
            if args.model == 'ResNet20_Q':
                not_layer = [6,12]
                if idx in not_layer :
                    continue
                else:
                    if param.shape[1] == 16:
                        image = 32
                    elif param.shape[1] == 32:
                        image = 16
                    elif param.shape[1] == 64:
                        image = 8

            elif args.model == 'WRN16-4_Q':
                not_layer = [0,4,8]
                if idx in not_layer :
                    continue
                else:
                    if param.shape[1] == 64:
                        image = 32
                    elif param.shape[1] == 128:
                        image = 16
                    elif param.shape[1] == 256:
                        image = 8

            _, pwr, _= SDK(image, image, param.shape[2], param.shape[2], param.shape[1], param.shape[0], args.ar, args.ac)
            weight_numpy = param.detach().cpu().numpy()
            GA_list_ = find_GA_v1(weight_numpy, pwr, args)
            GA_list.append(GA_list_)
            idxx += 1

    return GA_list

def find_GA_v1(arr, pw, args):   
    if arr.shape[2] == 3:
        cpy_arr = arr.copy()
        num_pattern = GA_channels_v1(pw, cpy_arr, args)
        num_pattern = num_pattern.astype(int)

    return num_pattern.tolist()

def GA_channels_v1(PW, param, args):

    skipp = 0
    ic = param.shape[1]
    best = 0

                
    if args.model == "WRN16-4_Q":
        if ic == 64:
            skipp = args.skipp[0]
            if args.mask == 6:
                if skipp <= 32:
                    ratio = 0.05
                elif 32 < skipp and skipp <= 64:
                    ratio = 0.1
                elif 64 < skipp and skipp <= 128:
                    ratio = 0.2
                elif 128 < skipp and skipp <= 512:
                    ratio = 0.3
                else:
                    ratio = 0.35

        elif ic == 128:
            skipp = args.skipp[1]
            if args.mask == 6:
                if skipp <= 64:
                    ratio = 0.05
                elif 64 < skipp and skipp <= 128:
                    ratio = 0.1
                elif 128 < skipp and skipp <= 256:
                    ratio = 0.2
                elif 256 < skipp and skipp <= 1024:
                    ratio = 0.3
                else:
                    ratio = 0.35
        elif ic == 256:
            skipp = args.skipp[2]  
            if args.mask == 6:
                if skipp <= 128:
                    ratio = 0.05
                elif 128 < skipp and skipp <= 256:
                    ratio = 0.1
                elif 256 < skipp and skipp <= 512:
                    ratio = 0.2
                elif 512 < skipp and skipp <= 2048:
                    ratio = 0.3
                else:
                    ratio = 0.35

    elif args.model == "ResNet20_Q":
        if ic == 16:
            skipp = args.skipp[0]
            if args.mask == 6:
                if skipp <= 32:
                    ratio = 0.2
                else:
                    ratio = 0.4
        elif ic == 32:
            skipp = args.skipp[1]
            if args.mask == 6:
                if skipp <= 64:
                    ratio = 0.2
                else:
                    ratio = 0.4
        elif ic == 64:
            skipp = args.skipp[2] 
            if args.mask == 6:
                if skipp <= 128:
                    ratio = 0.2
                else:
                    ratio = 0.4


    def f(X):
        nonlocal best

        if not ((np.sum(X) <= ic) and ((4*PW-4)*X[0]+(3*PW-2)*X[1]+(2*PW-1)*X[2]+PW*X[3]+X[4] == skipp)):
            importance_rate = 0
            rate = 1e-3

        else: 
            total_importance = param.shape[1]*param.shape[2]*param.shape[3]
            importance_p = X[0] + 2*X[1] + 4*X[2] + 6*X[3] + 8*X[4] + 9*X[5]
            
            importance_rate = importance_p / total_importance

            if importance_rate > best:
                best = importance_rate
                print(X)
                rate = 1
            else:
                rate = 1e-3
 
        return -(np.sum(X)*rate)
    

    varbound=np.array([[0,int(ic*ratio)]]*6)

    algorithm_param = {'max_num_iteration': 2000,\
                    'population_size':1000,\
                    'mutation_probability':0.1,\
                    'elit_ratio': 0.01,\
                    'crossover_probability': 0.5,\
                    'parents_portion': 0.3,\
                    'crossover_type':'uniform',\
                    'max_iteration_without_improv':None}


    model_=ga(function=f, dimension=6, variable_type='int', variable_boundaries=varbound, algorithm_parameters=algorithm_param)

    
    model_.run() 

    
    output = model_.result['variable']

    # print(output)
    return output




class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super().__init__()
        self.lambd = lambd
    def forward(self, x):
        return self.lambd(x)

class BasicBlock_Q(nn.Module):
    expansion = 1
    def __init__(self, a_bit, w_bit, in_planes, planes, stride=1, option='A'):
        super().__init__()
        self.w_bit = w_bit
        self.a_bit = a_bit
        self.act1 = Activate(self.a_bit)

        self.conv1 = Conv2d_Q_mask(self.w_bit, in_planes, planes, kernel_size=3, padding=1, stride=stride, bias=False)
        self.bn1 = SwitchBatchNorm2d(self.w_bit, planes)
        self.conv2 = Conv2d_Q_mask(self.w_bit, planes, planes, kernel_size=3, padding=1, stride=1, bias=False)
        self.bn2 = SwitchBatchNorm2d(self.w_bit, planes)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                '''
                For CIFAR10 ResNet paper uses option A.
                '''
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                    #Conv2d_Q_(self.w_bit, in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                    #SwitchBatchNorm2d(self.w_bit, self.expansion * planes)
                    ## Full-precision
                    nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion * planes)
                )
        self.act2 = Activate(self.a_bit)

    def forward(self, x):
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)  # x used here
        out = self.act2(out)
        return out

# ResNet code modified from original of [https://github.com/akamaster/pytorch_resnet_cifar10/blob/master/resnet.py]
# Modified version for our experiment.
class ResNet20_Q(nn.Module):
    def __init__(self, a_bit, w_bit, block, num_blocks, num_classes=10, expand=1): 
        super().__init__()
        self.in_planes = 16 # Resnet

        self.w_bit = w_bit
        self.a_bit = a_bit
        self.act = Activate(self.a_bit)

        self.layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1, stride=1, bias=False),
            SwitchBatchNorm2d(self.w_bit, 16),
            Activate(self.a_bit),
            
            *self._make_layer(block, 16, num_blocks[0], stride=1),
            *self._make_layer(block, 32, num_blocks[1], stride=2),
            *self._make_layer(block, 64, num_blocks[2], stride=2),
        )

        # mask_prune(self.layers)
        self.fc = nn.Linear(64, num_classes) 

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            # Full precision
            # option is 'A': Use F.pad
            # option is 'B': Use Conv+BN
            layers.append(block(self.a_bit, self.w_bit, self.in_planes, planes, stride, option='B'))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.layers(x)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out 



class Wide_BasicBlock_Q(nn.Module):
    expansion = 1
    def __init__(self, a_bit, w_bit, in_planes, planes, stride=1):
        super().__init__()
        self.w_bit = w_bit
        self.a_bit = a_bit
        self.act1 = Activate(self.a_bit)
        self.act2 = Activate(self.a_bit)

        self.dropout = nn.Dropout(0.3) # p = 0.3
        
        # conv2d_Q_ 바꿨음 나중에 수정해서 사용하기
        self.conv1 = Conv2d_Q_mask(self.w_bit, in_planes, planes, kernel_size=3, padding=(1,1), stride=stride, bias=False)
        self.bn1 = SwitchBatchNorm2d(self.w_bit, in_planes)
        self.conv2 = Conv2d_Q_mask(self.w_bit, planes, planes, kernel_size=3, padding=(1,1), stride=1, bias=False) 
        self.bn2 = SwitchBatchNorm2d(self.w_bit, planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
                self.shortcut = nn.Sequential(
                    # Conv2d_Q_(self.w_bit, in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                    nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                )

    def forward(self, x):
        x = self.act1(self.bn1(x))
        out = self.dropout(self.conv1(x))
        out = self.conv2(self.act2(self.bn2(out)))
        out += self.shortcut(x)  # x used here
        return out


class Wide_ResNet_Q(nn.Module):
    def __init__(self, a_bit, w_bit, block, num_blocks, scale, num_classes=10): 
        super().__init__()

        self.in_planes = 16
        self.w_bit = w_bit
        self.a_bit = a_bit
        self.act = Activate(self.a_bit)
        nStages = [16, 16*scale, 32*scale, 64*scale]
        self.bn1 = SwitchBatchNorm2d(self.w_bit, nStages[3])
        
        self.layers = nn.Sequential(
            nn.Conv2d(3, nStages[0], kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(self.in_planes),
            nn.ReLU(),
            
            *self._make_layer(block, nStages[1], num_blocks[0], stride=1), 
            *self._make_layer(block, nStages[2], num_blocks[1], stride=2),
            *self._make_layer(block, nStages[3], num_blocks[2], stride=2),
        )

        # mask_prune(self.layers)
        self.fc = nn.Linear(nStages[3], num_classes) 

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            # Full precision
            # option is 'A': Use F.pad
            # option is 'B': Use Conv+BN
            layers.append(block(self.a_bit, self.w_bit, self.in_planes, planes, stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.layers(x)
        out = self.act(self.bn1(out))
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out 



idxxx = -1
class Conv2d_Q_mask(Conv2d_Q): ## original
    def __init__(self, w_bit, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False):
        super(Conv2d_Q_mask, self).__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.w_bit = w_bit
        self.quantize_fn = weight_quantize_fn(self.w_bit)
        self.stride = stride
        

    def forward(self, X):
        global idxxx
        weight_q = self.quantize_fn(self.weight) 


        if args.model == 'WRN16-4_Q':
            if (self.stride == 1) and (self.weight.size()[1] != 16) and (self.weight.size()[2]==3):
                # print(self.weight.size())
                if idxxx == 8 :
                    idxxx = -1    

                idxxx += 1
                weight_q = weight_q * args.masked_pattern[idxxx]

        else:
            if (self.stride == 1) and (self.weight.size()[1] != 3) and (self.weight.size()[2]==3):
                # print(self.weight.size())
                if idxxx == 15 :
                    idxxx = -1    

                idxxx += 1
                # print(args.masked_pattern[idxxx].size())
                # print(weight_q.size())
                weight_q = weight_q * args.masked_pattern[idxxx]      
            # print(weight_q)

    
        return F.conv2d(X, weight_q, self.bias, self.stride, self.padding, self.dilation, self.groups)
    


parser = argparse.ArgumentParser(description='Pytorch PatDNN training')
parser.add_argument('--model',      default='WRN16-4_Q',          help = 'select model : ResNet20_Q, WRN16-4_Q')
parser.add_argument('--dir',        default='/Data',           help='dataset root')
parser.add_argument('--dataset',    default='cifar100',          help='select dataset')
parser.add_argument('--batchsize',  default=512, type=int,      help='set batch size')
parser.add_argument('--lr',         default=1e-3, type=float,   help='set learning rate') 
parser.add_argument('--rho',        default=6e-1, type=float,   help='set rho') # original 6e-1?
parser.add_argument('--connect_perc',  default=1, type=float, help='connectivity pruning ratio')
parser.add_argument('--epoch',      default=0, type=int,      help='set epochs') # 60
parser.add_argument('--exp',        default='test', type=str,   help='test or not')
parser.add_argument('--l2',         default=False, action='store_true', help='apply l3 regularization')
parser.add_argument('--scratch',    default=False, action='store_true', help='start from pretrain/scratch')
parser.add_argument('--no-cuda',    default=False, action='store_true', help='disables CUDA training')
parser.add_argument('--GPU', type=int, default=1) 
parser.add_argument('--ar', type=int, default=512)
parser.add_argument('--ac', type=int, default=256)
parser.add_argument('--ab', type=int, default=32)
parser.add_argument('--wb', type=int, default=2)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--mode', type=int, default=4, help='0: No-pruning, 1: Ours, 2: patdnn, 3: random, 4: pairs')
parser.add_argument('--withoc', type=int, default=2) 
parser.add_argument('--num_sets',   default='4', type=int,      help='# of pattern sets')
parser.add_argument('--workers', type=int, default=8, help = '')
parser.add_argument('--mask', type=int, default=2, help = '1,2,3,4,5')
parser.add_argument('--rows', type=int, default=0)
parser.add_argument('--percentage', type=int, default=0)
args = parser.parse_args()
print(args)


print(f'Actual used array columns = {args.ac}')

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

if args.mode == 0:
    name = 'No_pruning'
elif args.mode == 1:
    name = 'Adaptive_PAIRS_fixed' # row 개수 patdnn과 동일
elif args.mode == 2:
    name = 'PATDNN'
elif args.mode == 3:
    name = 'RANDOM'
elif args.mode == 4:
    name = 'PAIRS'
elif args.mode == 5:
    name = 'Adaptive_PAIRS_variablerows'


directory_save = './log/%s/%s/%s/ab%s_wb%s/%s'%(args.model, args.dataset, args.seed, args.ab, args.wb, name)
if not os.path.isdir(directory_save):
    os.makedirs(directory_save)

if args.mode == 5:
    file_name = directory_save + '/%dx%d_rows%d.log'%(args.ar, args.ac, args.rows)
else :
    file_name = directory_save + '/%dx%d_mask%d.log'%(args.ar, args.ac, args.mask)


if args.model == "ResNet20_Q":
    if args.mask == 1:
        args.skipp = [16, 32, 64]
    elif args.mask == 2:
        args.skipp = [32, 64, 128]
    elif args.mask == 3:
        args.skipp = [80, 128, 256]
    elif args.mask == 4:
        args.skipp = [96, 160, 320]
    elif args.mask == 5:
        args.skipp = [144, 224, 448]
    elif args.mask == 6:
        args.skipp = [int(args.rows/7), int(args.rows*2/7), int(args.rows*4/7)]

else :
    if args.mask == 1:
        args.skipp = [64, 128, 256]
    elif args.mask == 2:
        args.skipp = [128, 256, 512]
    elif args.mask == 3:
        args.skipp = [320, 512, 1024]
    elif args.mask == 4:
        args.skipp = [384, 640, 1280]
    elif args.mask == 5:
        args.skipp = [576, 896, 1792]
    elif args.mask == 6:
        args.skipp = [int(args.rows/7), int(args.rows*2/7), int(args.rows*4/7)]

file_handler = logging.FileHandler(file_name)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


# pt_save = './pt_save/%s/%s'%(args.model, args.dataset)
print('new model uploaded...')


##### Load Dataset ####
print('\nPreparing Dataset...')
train_loader, test_loader = data_loader(args.dir, args.dataset, args.batchsize, args.workers)
print('Dataset Loaded')

mask_dic = './pattern_set'
if not os.path.isdir(directory_save):
    os.makedirs(directory_save)

print('\nDefined Model is loaded...')
if args.model == 'ResNet20_Q' :
    model = ResNet20_Q(args.ab, args.wb, block=BasicBlock_Q, num_blocks=[3,3,3], num_classes=10).cuda()

elif args.model == 'WRN16-4_Q' :
    model = Wide_ResNet_Q(args.ab, args.wb, block=Wide_BasicBlock_Q, num_blocks=[2,2,2], scale=4, num_classes=100).cuda()



model_save = './save'
print('pretrained model uploaded...')
if args.model == 'ResNet20_Q' :
    model.load_state_dict(torch.load(model_save+'/ResNet20/lr0.1000_wb2_ab32.pt'), strict=False)

elif args.model == 'WRN16-4_Q' :
    model.load_state_dict(torch.load(model_save+'/WRN16-4/lr0.1000_wb2_ab32.pt'), strict=False)


print('Patternset is loaded...')

if args.mode == 0:
    pattern_set = {
                        "pattern_0": np.array([[1,1,1,1,1,1,1,1,1]]),
                        "pattern_1": np.array([[1,1,1,1,1,1,1,1,1]]),
                        
                        "pattern_2": np.array([[1,1,1,1,1,1,1,1,1]]),
                        
                        "pattern_3": np.array([[1,1,1,1,1,1,1,1,1]]),
                        "pattern_4": np.array([[1,1,1,1,1,1,1,1,1]]),
                        "pattern_5": np.array([[1,1,1,1,1,1,1,1,1]])
                    }
                    
elif args.mode == 1 or args.mode == 5:
    pattern_set = { # 1, 2, 4, 6, 8 entry를 이용
                        # "pattern_0": np.array([[0,0,0,0,0,0,0,0,0]]),

                        "pattern_0": np.array([[0,0,0,0,1,0,0,0,0]]),

                        "pattern_1": np.array([[0,0,0,0,1,0,0,1,0], [0,1,0,0,1,0,0,0,0], [0,0,0,0,1,1,0,0,0], [0,0,0,1,1,0,0,0,0]]),

                        "pattern_2": np.array([[0,0,0,0,1,1,0,1,1], [0,0,0,1,1,0,1,1,0], [1,1,0,1,1,0,0,0,0], [0,1,1,0,1,1,0,0,0]]),
                        
                        "pattern_3": np.array([[0,0,0,1,1,1,1,1,1], [1,1,1,1,1,1,0,0,0], [1,1,0,1,1,0,1,1,0], [0,1,1,0,1,1,0,1,1]]),
                        
                        "pattern_4": np.array([[0,1,1,1,1,1,1,1,1], [1,1,0,1,1,1,1,1,1], [1,1,1,1,1,1,0,1,1], [1,1,1,1,1,1,1,1,0]]),
                        
                        "pattern_5": np.array([[1,1,1,1,1,1,1,1,1]])
                    }
    
elif args.mode == 2 :
    if os.path.isfile(directory_save + '/mask_' + str(args.mask) + '.npy') is False:
        pattern_set = pattern_setter(model, args.mask)
        np.save(directory_save + '/mask_' + str(args.mask) + '.npy', pattern_set)
    else:
        pattern_set = np.load(directory_save + '/mask_' + str(args.mask) + '.npy')

    pattern_set = pattern_set[:args.num_sets, :]

elif args.mode == 3:
    candi_list = []
    ran_list = [0,1,2,3,4,5,6,7,8]
    for i in range(args.num_sets) :
        one_list = [1,1,1,1,1,1,1,1,1]
        sample = random.sample(ran_list, args.mask)
        for sam in sample :
            one_list[sam] = 0
        candi_list.append(one_list)
    
    pattern_set = np.array(candi_list)


elif args.mode == 4:
    if args.mask == 5 : # 4
        pattern_set = np.array([[0,0,0,0,1,1,0,1,1], [0,0,0,1,1,0,1,1,0], [1,1,0,1,1,0,0,0,0], [0,1,1,0,1,1,0,0,0]])
    elif args.mask == 3 : # 4
        pattern_set = np.array([[0,0,0,1,1,1,1,1,1], [1,1,1,1,1,1,0,0,0], [1,1,0,1,1,0,1,1,0], [0,1,1,0,1,1,0,1,1]])
    elif args.mask == 1: # 4
        pattern_set = np.array([[0,1,1,1,1,1,1,1,1], [1,1,0,1,1,1,1,1,1], [1,1,1,1,1,1,0,1,1], [1,1,1,1,1,1,1,1,0]])
    elif args.mask == 2 : # 14
        pattern_set = np.array([
                                [0,0,1,1,1,1,1,1,1], [1,0,0,1,1,1,1,1,1], [0,1,1,0,1,1,1,1,1], [1,1,0,1,1,0,1,1,1],
                                [1,1,1,0,1,1,0,1,1], [1,1,1,1,1,1,0,0,1], [1,1,1,1,1,1,1,0,0], [1,1,1,1,1,0,1,1,0],
                                [0,1,0,1,1,1,1,1,1], [1,1,0,1,1,1,1,1,0], [1,1,1,1,1,1,0,1,0], [0,1,1,1,1,1,0,1,1],
                                [0,1,1,1,1,1,1,1,0], [1,1,0,1,1,1,0,1,1]
                                ])
    elif args.mask == 4 :# 16
        pattern_set = np.array([
                                [0,0,0,0,1,1,1,1,1], [0,0,0,1,1,0,1,1,1], [0,0,0,1,1,1,0,1,1], [0,0,0,1,1,1,1,1,0],
                                [1,1,1,1,1,0,0,0,0], [1,1,1,0,1,1,0,0,0], [0,1,1,1,1,1,0,0,0], [1,1,0,1,1,1,0,0,0],
                                [0,1,1,0,1,1,0,0,1], [0,0,1,0,1,1,0,1,1], [0,1,0,0,1,1,0,1,1], [0,1,1,0,1,1,0,1,0],
                                [1,1,0,1,1,0,1,0,0], [1,0,0,1,1,0,1,1,0], [0,1,0,1,1,0,1,1,0], [1,1,0,1,1,0,0,1,0]
                                ])


print('\nPatternset is uploaded...')
args.pattern_set = pattern_set
print(pattern_set)

print('*'*100)
logger.info(f"Pattern set : {pattern_set}")


start = time.time() # 시작

if args.model == 'ResNet20_Q':
    dic_path = '/workspace/GAROS/log/ResNet20_Q/json'
else:
    dic_path = '/workspace/GAROS/log/WRN16-4_Q/json'

if args.mode == 0 or args.mode == 1:
    if os.path.isfile(dic_path + '/mask%d_0901'%(args.mask) + '.json') is False:
        GA_list = GA_selction_v1(model, args)
        with open(dic_path + '/mask%d_0901.json'%(args.mask), 'w') as f :
            json.dump(GA_list, f, indent=4)
    else:
        with open(dic_path + '/mask%d_0901.json'%(args.mask)) as f:
            GA_list = json.load(f)

    logger.info(GA_list)

elif args.mode == 5 or args.mode == 6:
    if os.path.isfile(dic_path + '/rows%d'%(args.rows) + '.json') is False:
        GA_list = GA_selction_v1(model, args)
        with open(dic_path + '/rows%d.json'%(args.rows), 'w') as f :
            json.dump(GA_list, f, indent=4)
    else:
        with open(dic_path + '/rows%d.json'%(args.rows)) as f:
            GA_list = json.load(f)


# print(GA_list)
sec = time.time()-start # 종료 - 시작 (걸린 시간)
 
times = str(datetime.timedelta(seconds=sec)) # 걸린시간 보기좋게 바꾸기
short = times.split(".")[0] # 초 단위 까지만
logger.info("GA operation time")
logger.info(f"{short} sec")


print("Retraining for fine tuning...")
logger.info("="*100)
logger.info("Retraining for fine tuning...")







def cos_sim(A, B):
  vector_A = A.flatten()
  vector_B = B.flatten()
  return dot(vector_A, vector_B)/(norm(vector_A)*norm(vector_B))

def dist(x,y):
  return np.sqrt(((x-y)**2).sum())
  # return np.sqrt(np.sum((x-y)**2))

# Real Pruning ! ! !
print("\nApply Pruning with connectivity & pattern set...")
# print(pattern_loss)
def apply_prune_pat_v1(args, model, device, pattern_set, GA_list):
    # dict_mask = {}
    list_mask = []
    idx = -1
    idxx = 0

    quantize_fn = weight_quantize_fn(2)

    for name, param in model.named_parameters():
        if name.split('.')[-1] == "weight" and len(param.shape) == 4 and 'downsample' not in name and 'conv' in name:
            # logger.info("="*50)
            # logger.info(name)
            # logger.info(param.size())
            idx += 1
            if args.model == 'ResNet20_Q':
                not_layer = [6,12]
                if idx in not_layer :
                    continue
                else:
                    if param.shape[1] == 16:
                        image = 32
                    elif param.shape[1] == 32:
                        image = 16
                    elif param.shape[1] == 64:
                        image = 8

            elif args.model == 'WRN16-4_Q':
                not_layer = [0,4,8]
                if idx in not_layer :
                    continue
                else:
                    if param.shape[1] == 64:
                        image = 32
                    elif param.shape[1] == 128:
                        image = 16
                    elif param.shape[1] == 256:
                        image = 8

            param__ = param.clone()
            # param__v = quantize_fn(param__).clone # 기존에는 이코드가 없었음 240504
            param__ = param__.cpu().detach().numpy()

            ## 이 부분 추가 ###
            # if args.model == 'ResNet20_Q':
            #     if idx == 5 or idx == 11 or idx == 17:
            #         np.save(directory_save + 'original_mask%d_%d.npy'%(args.mask, idx), param__)
            # else:
            #     if idx == 1 or idx == 6 or idx == 10: # 123, 567, 91011
            #         np.save(directory_save + 'original_mask%d_%d.npy'%(args.mask, idx), param__)              
            ## ###

            _, pwr, _= SDK(image, image, param.shape[2], param.shape[2], param.shape[1], param.shape[0], args.ar, args.ac)

            if args.mode == 0 or args.mode == 1 or args.mode == 5:
                mask = prune_weight(param__, device, args.connect_perc, pattern_set, pwr, GA_list[idxx], args)
                '''
                mask shape 확인용
                '''
                # if param.shape[1] == 16:
                #     np.save(directory_save + '/ours_mask%d.npy'% (args.mask), mask)

            else:
                mask = prune_weight_p(param__, device, args.connect_perc, pattern_set, args)

                # if param.shape[1] == 16:
                    # np.save(directory_save + '/pairs_mask%d.npy'% (args.mask), mask)
      
            param__v = quantize_fn(param.clone()) # 기존에는 이코드가 없었음 240504
            param_ = param__v.cpu().detach().numpy() * mask
            if args.model == 'ResNet20_Q':
                # if idx == 4: #5 or idx == 11 or idx == 17:
                # print(name)
                # #     print(param_.shape)
                rows = counting(mask, mask.shape, 4,4)
                print(rows)
                    # logger.info(cos_sim(param_, param__))
                    # np.save(directory_save + 'mask%d_%d.npy'%(args.mask, idx), mask)
                    
                    # for i in range(len(param__[1])):
                    #     # print('*'*10)
                    #     print(np.sum(np.abs(param__[:, i, :, :])))
                    #     print(np.count_nonzero(mask[0, i]))
                    #     print(np.sum(np.abs(param_[:, i, :, :])))
                # print(np.sum(np.abs(param_)))
            else:
                # print(idx, name)
                if idx == 1 or idx == 6 or idx == 10: # 123, 567, 91011
                    # logger.info(cos_sim(param_, param__))
                    np.save(directory_save + 'mask%d_%d.npy'%(args.mask, idx), mask)      
                    
            # diff = param__ - param_
            # loss += np.linalg.norm(diff)
            # logger.info(np.linalg.norm(param__- param_))
            # rows = counting(mask, mask.shape, pwr, pwr)
            # logger.info('# of skipped rows')
            # logger.info(rows)
            mask = torch.Tensor(param_ != 0).to(device)
            # dict_mask[name] = mask
            list_mask.append(mask)
            idxx += 1
        
    # return dict_mask
    return list_mask


# mask = apply_prune_pat_v1(args, model, device, pattern_set, GA_list)


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


def train_model(args, model, train_loader, test_loader):
    logger.info("="*100)
    best_acc = -1
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 20, 1e-4)

    if args.mode == 0 or args.mode == 1 or args.mode == 5:
        args.masked_pattern = apply_prune_pat_v1(args, model, device, pattern_set, GA_list)
    else :
        args.masked_pattern = apply_prune_pat_v1(args, model, device, pattern_set, None)

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
        
        # loss_sum = loss_sum  / cnt
        loss_sum = (loss_sum / cnt) 
        model.eval()
        acc = eval(model, test_loader)
        print(f'Epochs : {epoch+1}, Accuracy : {acc}')
        logger.info("Epoch %d/%d, Acc=%.4f"%(epoch+1, args.epoch, acc))
        
        scheduler.step()

        if acc > best_acc :
            best_acc = acc
            # if args.mode == 0 or args.mode == 1 or args.mode == 5:
            #     args.masked_pattern = apply_prune_pat_v1(args, model, device, pattern_set, GA_list)
            print('Best accuracy is updated! at epoch %d/%d: %.4f '%(epoch, args.epoch, best_acc))
            logger.info('Best accuracy is : %.4f '%(best_acc))
    
    print('Final best accuracy is : %.4f '%(best_acc))
    logger.info('Final best accuracy is : %.4f '%(best_acc))




train_model(args, model, train_loader, test_loader)
