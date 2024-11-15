from re import L
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from torch.nn.modules.batchnorm import BatchNorm2d
import copy
from utils_1 import *
import function



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

        self.conv1 = Conv2d_Q_(self.w_bit, in_planes, planes, kernel_size=3, padding=1, stride=stride, bias=False)
        self.bn1 = SwitchBatchNorm2d(self.w_bit, planes)
        self.act1 = Activate(self.a_bit)

        self.conv2 = Conv2d_Q_(self.w_bit, planes, planes, kernel_size=3, padding=1, stride=1, bias=False)
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
    


        
'''
Quantization Network
'''
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

        self.conv1 = Conv2d_Q_(self.w_bit, in_planes, planes, kernel_size=3, padding=1, stride=stride, bias=False)
        self.bn1 = SwitchBatchNorm2d(self.w_bit, planes)
        self.conv2 = Conv2d_Q_(self.w_bit, planes, planes, kernel_size=3, padding=1, stride=1, bias=False)
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
        self.conv1 = Conv2d_Q_(self.w_bit, in_planes, planes, kernel_size=3, padding=(1,1), stride=stride, bias=False)
        self.bn1 = SwitchBatchNorm2d(self.w_bit, in_planes)
        self.conv2 = Conv2d_Q_(self.w_bit, planes, planes, kernel_size=3, padding=(1,1), stride=1, bias=False) 
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
    
