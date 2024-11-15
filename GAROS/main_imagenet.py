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
from geneticalgorithm2.geneticalgorithm2 import geneticalgorithm2 as ga

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal




def GA_selction_v1(model, args):
    idx = -1
    idxx = 0
    GA_list = []

    for name, param in model.named_parameters():
        if name.split('.')[-1] == "weight" and len(param.shape) == 4 and 'downsample' not in name and 'conv' in name:
            idx += 1
            if args.model == 'ResNet18_Q':
                not_layer = [0, 5, 9, 13]
                if idx in not_layer :
                    continue
                else:
                    if param.shape[1] == 64:
                        image = 56
                    elif param.shape[1] == 128:
                        image = 28
                    elif param.shape[1] == 256:
                        image = 14
                    else:
                        image = 7

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

    if args.mask == 1 or args.mask == 2:
        ratio = 0.2
    else :
        ratio = 0.4

    if args.model == "ResNet18_Q":
        if ic == 64:
            skipp = args.skipp[0]
            if skipp <= 64:
                ratio = 0.1
            # elif 32 < skipp and skipp <= 64:
            #     ratio = 0.1
            elif 64 < skipp and skipp <= 128:
                ratio = 0.2
            else:
                ratio = 0.3
            
        elif ic == 128:
            skipp = args.skipp[1]
            if skipp <= 128:
                ratio = 0.1
            # elif 64 < skipp and skipp <= 128:
            #     ratio = 0.1
            elif 128 < skipp and skipp <= 256:
                ratio = 0.2
            else:
                ratio = 0.3

        elif ic == 256:
            skipp = args.skipp[2]  
            if skipp <= 256:
                ratio = 0.1
            # elif 128 < skipp and skipp <= 256:
            #     ratio = 0.1
            elif 256 < skipp and skipp <= 512:
                ratio = 0.2
            else:
                ratio = 0.3

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

    # print(model.result['variable'])
    # if ic-np.sum(model.result['variable']) <= 0:
    #     resi = 0
    # else :
    #     resi = ic-np.sum(model.result['variable'])
    
    # output = np.append(model.result['variable'], resi) # no pruning을 추가해주기 위해
    
    output = model_.result['variable']
    if ic - np.sum(output) != 0:
        output[-1] = output[-1] + (ic-np.sum(output))
    # print(output)
    return output



def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, a_bit, w_bit, in_planes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()

        self.w_bit = w_bit
        self.a_bit = a_bit


        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1

        self.conv1 = Conv2d_Q_mask(self.w_bit, in_planes, planes, kernel_size=3, padding=1, stride=stride, bias=False)
        self.bn1 = SwitchBatchNorm2d(self.w_bit, planes)
        self.act1 = Activate(self.a_bit)

        self.conv2 = Conv2d_Q_mask(self.w_bit, planes, planes, kernel_size=3, padding=1, stride=1, bias=False)
        self.bn2 = SwitchBatchNorm2d(self.w_bit, planes)
        self.act2 = Activate(self.a_bit)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.act2(out)

        return out

class ResNet18_Q(nn.Module):

    def __init__(self, block, layers, a_bit, w_bit, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet18_Q, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        self.a_bit = a_bit
        self.w_bit = w_bit
        

        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                # if isinstance(m, Bottleneck):
                #     nn.init.constant_(m.bn3.weight, 0)
                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),)

        layers = []
        layers.append(block(self.a_bit, self.w_bit, self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.a_bit, self.w_bit, self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def forward(self, x):
        return self._forward_impl(x)
    

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

        if (self.stride == 1) and (self.weight.size()[1] != 3) and (self.weight.size()[2]==3):
            if idxxx == 12:
                idxxx = -1

            idxxx += 1
            weight_q = weight_q * args.masked_pattern[idxxx]

    
        return F.conv2d(X, weight_q, self.bias, self.stride, self.padding, self.dilation, self.groups)
    


parser = argparse.ArgumentParser(description='Pytorch PatDNN training')
parser.add_argument('--model',      default='ResNet18_Q',          help = 'select model : ResNet18_Q')
parser.add_argument('--dir',        default='/Data',           help='dataset root')
parser.add_argument('--dataset',    default='imagenet',          help='select dataset')
parser.add_argument('--batchsize',  default=512, type=int,      help='set batch size')
parser.add_argument('--lr',         default=1e-3, type=float,   help='set learning rate') 
parser.add_argument('--rho',        default=6e-1, type=float,   help='set rho') # original 6e-1?
parser.add_argument('--connect_perc',  default=1, type=float, help='connectivity pruning ratio')
parser.add_argument('--epoch',      default=10, type=int,      help='set epochs') # 60
parser.add_argument('--exp',        default='test', type=str,   help='test or not')
parser.add_argument('--l2',         default=False, action='store_true', help='apply l3 regularization')
parser.add_argument('--scratch',    default=False, action='store_true', help='start from pretrain/scratch')
parser.add_argument('--no-cuda',    default=False, action='store_true', help='disables CUDA training')
parser.add_argument('--GPU', type=int, default=2) 
parser.add_argument('--ar', type=int, default=2048)
parser.add_argument('--ac', type=int, default=1024)
parser.add_argument('--ab', type=int, default=32)
parser.add_argument('--wb', type=int, default=4)
parser.add_argument('--seed', type=int, default=4)
parser.add_argument('--mode', type=int, default=1, help='0: No-pruning, 1: Ours, 2: patdnn, 3: random, 4: pairs')
parser.add_argument('--withoc', type=int, default=2) 
parser.add_argument('--num_sets',   default='4', type=int,      help='# of pattern sets')
parser.add_argument('--workers', type=int, default=16, help = '')
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
    name = 'Acordion_fix' # row 개수 patdnn과 동일
elif args.mode == 2:
    name = 'PATDNN'
elif args.mode == 3:
    name = 'RANDOM'
elif args.mode == 4:
    name = 'PAIRS_no_retraining'
elif args.mode == 5:
    name = 'Acordion_variable'
elif args.mode == 6:
    name = 'Acordion_variable_re'


directory_save = './log/%s/%s/%s/ab%s_wb%s/%.4f/%s'%(args.model, args.dataset, args.seed, args.ab, args.wb, args.lr, name)
if not os.path.isdir(directory_save):
    os.makedirs(directory_save)

if args.mode == 5 or args.mode == 6:
    file_name = directory_save + '/%dx%d_rows%d.log'%(args.ar, args.ac, args.rows)
else :
    file_name = directory_save + '/%dx%d_mask%d.log'%(args.ar, args.ac, args.mask)

'''
Variable 코드
'''


if args.model == "ResNet18_Q":
    if args.mask == 1:
        args.skipp = [64, 128, 256, 0]
    elif args.mask == 2:
        args.skipp = [128, 256, 512, 0]
    elif args.mask == 3:
        args.skipp = [320, 512, 1024, 0]
    elif args.mask == 4:
        args.skipp = [384, 640, 1280, 0]
    elif args.mask == 5:
        args.skipp = [576, 896, 1792, 0]


    elif args.mask == 6:
        # if args.rows <= 2560:
        #     divide = int(args.rows // 512)
        #     resi = int(args.rows % 512)
        # else:
        #     divide = 5
        #     resi = args.rows - 512*divide

        # if args.rows <= 512:
        #     args.mask = 0
        # else:
        #     args.mask = divide
        args.skipp = [int(args.rows/7), int(args.rows*2/7), int(args.rows*4/7), 0]


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
model = ResNet18_Q(BasicBlock, [2, 2, 2, 2], args.ab, args.wb, num_classes = 1000).cuda()

model_save = './save'
print('pretrained model uploaded...')
model.load_state_dict(torch.load(model_save+'/ResNet18/lr0.1000_wb4_ab32.pt'), strict=False)


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
                    
elif args.mode == 1 or args.mode == 5 or args.mode == 6:
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

dic_path = '/workspace/PAIRS_v1/log/ResNet18_Q/json'


if args.mode == 0 or args.mode == 1:
    if os.path.isfile(dic_path + '/mask%d_0901'%(args.mask) + '.json') is False:
        GA_list = GA_selction_v1(model, args)
        with open(dic_path + '/mask%d_0901.json'%(args.mask), 'w') as f :
            json.dump(GA_list, f, indent=4)
    else:
        with open(dic_path + '/mask%d_0901.json'%(args.mask)) as f:
            GA_list = json.load(f)

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

# Real Pruning ! ! !
print("\nApply Pruning with connectivity & pattern set...")
# print(pattern_loss)
def apply_prune_pat_v1(args, model, device, pattern_set, GA_list):
    # dict_mask = {}
    list_mask = []
    idx = -1
    idxx = 0
    for name, param in model.named_parameters():
        if name.split('.')[-1] == "weight" and len(param.shape) == 4 and 'downsample' not in name and 'conv' in name:
            # print(name, param.size())
            logger.info("="*50)
            logger.info(name)
            idx += 1
            not_layer = [0, 5, 9, 13]
            if idx in not_layer :
                continue
            else:
                if param.shape[1] == 64:
                    image = 56
                elif param.shape[1] == 128:
                    image = 28
                elif param.shape[1] == 256:
                    image = 14
                else:
                    image = 7

            # print(name, param.size(), idx)
            param__ = param.clone()
            param__ = param__.cpu().detach().numpy()
            _, pwr, _= SDK(image, image, param.shape[2], param.shape[2], param.shape[1], param.shape[0], args.ar, args.ac)

            if args.mode == 0 or args.mode == 1 or args.mode == 5 or args.mode == 6:
                if param.shape[1] != 512:
                    mask = prune_weight(param__, device, args.connect_perc, pattern_set, pwr, GA_list[idxx], args)
                else:
                    # if args.mask == 5 : # 4
                    #     pattern_set1 = np.array([[0,0,0,0,1,1,0,1,1], [0,0,0,1,1,0,1,1,0], [1,1,0,1,1,0,0,0,0], [0,1,1,0,1,1,0,0,0]])
                    # elif args.mask == 3 : # 4
                    #     pattern_set1 = np.array([[0,0,0,1,1,1,1,1,1], [1,1,1,1,1,1,0,0,0], [1,1,0,1,1,0,1,1,0], [0,1,1,0,1,1,0,1,1]])
                    # elif args.mask == 1: # 4
                    #     pattern_set1 = np.array([[0,1,1,1,1,1,1,1,1], [1,1,0,1,1,1,1,1,1], [1,1,1,1,1,1,0,1,1], [1,1,1,1,1,1,1,1,0]])
                    # elif args.mask == 2 : # 14
                    #     pattern_set1 = np.array([
                    #                             [0,0,1,1,1,1,1,1,1], [1,0,0,1,1,1,1,1,1], [0,1,1,0,1,1,1,1,1], [1,1,0,1,1,0,1,1,1],
                    #                             [1,1,1,0,1,1,0,1,1], [1,1,1,1,1,1,0,0,1], [1,1,1,1,1,1,1,0,0], [1,1,1,1,1,0,1,1,0],
                    #                             [0,1,0,1,1,1,1,1,1], [1,1,0,1,1,1,1,1,0], [1,1,1,1,1,1,0,1,0], [0,1,1,1,1,1,0,1,1],
                    #                             [0,1,1,1,1,1,1,1,0], [1,1,0,1,1,1,0,1,1]
                    #                             ])
                    # elif args.mask == 4 :# 16
                    #     pattern_set1 = np.array([
                    #                             [0,0,0,0,1,1,1,1,1], [0,0,0,1,1,0,1,1,1], [0,0,0,1,1,1,0,1,1], [0,0,0,1,1,1,1,1,0],
                    #                             [1,1,1,1,1,0,0,0,0], [1,1,1,0,1,1,0,0,0], [0,1,1,1,1,1,0,0,0], [1,1,0,1,1,1,0,0,0],
                    #                             [0,1,1,0,1,1,0,0,1], [0,0,1,0,1,1,0,1,1], [0,1,0,0,1,1,0,1,1], [0,1,1,0,1,1,0,1,0],
                    #                             [1,1,0,1,1,0,1,0,0], [1,0,0,1,1,0,1,1,0], [0,1,0,1,1,0,1,1,0], [1,1,0,1,1,0,0,1,0]
                    #                             ])
                    # elif args.mask == 0:
                    pattern_set1 = np.array([[1,1,1,1,1,1,1,1,1]])

                    mask = prune_weight_p(param__, device, args.connect_perc, pattern_set1, args)
                '''
                mask shape 확인용
                '''
            else:
                if param.shape[1] != 512:
                    mask = prune_weight_p(param__, device, args.connect_perc, pattern_set, args)
                else:
                    pattern_set1 = np.array([[1,1,1,1,1,1,1,1,1]])
                    mask = prune_weight_p(param__, device, args.connect_perc, pattern_set1, args)
                # if param.shape[1] == 16:
                #     np.save(directory_save + '/pairs_mask%d.npy'% (args.mask), mask)

            param_ = param__ * mask
            # diff = param__ - param_
            # loss += np.linalg.norm(diff)
            # logger.info(np.linalg.norm(param__- param_))
            rows = counting(mask, mask.shape, pwr, pwr)
            logger.info('# of skipped rows')
            logger.info(rows)
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
    T = 10
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T, 1e-4)

    if args.mode == 0 or args.mode == 1 or args.mode == 5 or args.mode == 6:
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



# train_model(args, model, train_loader, test_loader)
if args.mode == 0 or args.mode == 1 or args.mode == 5 or args.mode == 6:
    args.masked_pattern = apply_prune_pat_v1(args, model, device, pattern_set, GA_list)
else :
    args.masked_pattern = apply_prune_pat_v1(args, model, device, pattern_set, None)

model.eval()
acc = eval(model, test_loader)
print(f'Accuracy : {acc}')
logger.info("Acc=%.4f"%(acc))

    