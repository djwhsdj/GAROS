from re import L
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from torch.nn.modules.batchnorm import BatchNorm2d

import copy
# from geneticalgorithm2 import geneticalgorithm2 as ga

def counting (mask, layer_shape, pwr, pwh, mode = True) :
    # mask = mask.cpu().numpy()
    
    OC, IC, kr, kh = layer_shape

    cnt = 0

    kernel = []
    for i in range(kr*kh) :
        kernel.append(i)
    
    for i in range(IC) :
        pw = []
        for j in range(pwr*pwh) :
            pw.append([])
            
        for a in range(pwh-kh+1) :
            for b in range(pwr-kr+1) :
                for c in range(len(kernel)) :
                    divider = c // 3
                    residue = c % 3
                    pw_idx = (divider+a)*pwr+(residue+b)
                    pw[pw_idx].append(kernel[c])
        
        zero_list = []
        for j in range(kr) :
            for k in range(kh) :
                cal = mask[:, i, j, k].sum()
                if cal == 0 :
                    idx = j*kr + k
                    zero_list.append(idx)

        for q in range(len(pw)) :
            for j in zero_list :
                if j in pw[q] :
                    pw[q].remove(j)

        for m in pw :
            if m == [] :
                cnt+=1
                

    return cnt


class CrossEntropyLossSoft(torch.nn.modules.loss._Loss):
    """ inplace distillation for image classification.
    Refer to https://github.com/JiahuiYu/slimmable_networks/blob/master/utils/loss_ops.py
    """
    def forward(self, output, target):
        output_log_prob = torch.nn.functional.log_softmax(output, dim=1)
        target = target.unsqueeze(1)
        output_log_prob = output_log_prob.unsqueeze(2)
        cross_entropy_loss = -torch.bmm(target, output_log_prob)
        cross_entropy_loss = cross_entropy_loss.mean()
        return cross_entropy_loss
        
class Activate(nn.Module):
    def __init__(self, a_bit, quantize=True):
        super(Activate, self).__init__()
        self.abit = a_bit
        # Since ReLU is not differentible at x=0, changed to GELU
        #self.acti = nn.ReLU(inplace=True)
        self.acti = nn.GELU()
        self.quantize = quantize
        if self.quantize:
            self.quan = activation_quantize_fn(self.abit)

    def forward(self, x):
        if self.abit == 32:
            x = self.acti(x)
        else:
            x = torch.clamp(x, 0.0, 1.0)
        if self.quantize:
            x = self.quan(x)
        return x

class activation_quantize_fn(nn.Module):
    def __init__(self, a_bit):
        super(activation_quantize_fn, self).__init__()
        self.abit = a_bit
        assert self.abit <= 8 or self.abit == 32

    def forward(self, x):
        if self.abit == 32:
            activation_q = x
        else:
            activation_q = qfn.apply(x, self.abit)
        return activation_q


class qfn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, k):
        n = float(2**k - 1)
        out = torch.round(input * n) / n
        return out

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input, None


class weight_quantize_fn(nn.Module):
    def __init__(self, w_bit):
        super(weight_quantize_fn, self).__init__()
        self.wbit = w_bit
        assert self.wbit <= 8 or self.wbit == 32

    def forward(self, x):
        if self.wbit == 32:
            E = torch.mean(torch.abs(x)).detach()
            weight = torch.tanh(x)
            weight = weight / torch.max(torch.abs(weight))
            weight_q = weight * E
        else:
            # print(torch.abs(x))
            # pat = torch.where(torch.abs(x)>0, 1.0, 0.0)
            E = torch.mean(torch.abs(x)).detach()
            weight = torch.tanh(x)
            weight = weight / 2 / torch.max(torch.abs(weight)) + 0.5
            weight_q = 2 * qfn.apply(weight, self.wbit) - 1
            weight_q = weight_q * E
            # print(weight_q)
        return weight_q




class SwitchBatchNorm2d(nn.Module):
    """Adapted from https://github.com/JiahuiYu/slimmable_networks
    """
    def __init__(self, w_bit, num_features):
        super(SwitchBatchNorm2d, self).__init__()
        self.w_bit = w_bit
        self.bn_dict = nn.ModuleDict()
        # for i in self.bit_list:
        #     self.bn_dict[str(i)] = nn.BatchNorm2d(num_features)
        self.bn_dict[str(w_bit)] = nn.BatchNorm2d(num_features, eps=1e-4)

        self.abit = self.w_bit
        self.wbit = self.w_bit
        if self.abit != self.wbit:
            raise ValueError('Currenty only support same activation and weight bit width!')

    def forward(self, x):
        x = self.bn_dict[str(self.abit)](x)
        return x

class SwitchBatchNorm2d_(SwitchBatchNorm2d) : ## 만든거
    def __init__(self, w_bit, num_features) :
        super(SwitchBatchNorm2d_, self).__init__(num_features=num_features, w_bit=w_bit)
        self.w_bit = w_bit      
        # return SwitchBatchNorm2d_
    


def batchnorm2d_fn(w_bit):
    class SwitchBatchNorm2d_(SwitchBatchNorm2d):
        def __init__(self, num_features, w_bit=w_bit):
            super(SwitchBatchNorm2d_, self).__init__(num_features=num_features, w_bit=w_bit)

    return SwitchBatchNorm2d_


class Conv2d_Q(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super(Conv2d_Q, self).__init__(*args, **kwargs)


class Conv2d_Q_(Conv2d_Q): ## original
    def __init__(self, w_bit, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1,
                    bias=False):
        super(Conv2d_Q_, self).__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.w_bit = w_bit
        self.quantize_fn = weight_quantize_fn(self.w_bit)

    def forward(self, input):
        weight_q = self.quantize_fn(self.weight) 
        
        return F.conv2d(input, weight_q, self.bias, self.stride, self.padding, self.dilation, self.groups)



'''
def GA_selection_v1(weight, pwr, args):
    weight_numpy = weight.detach().cpu().numpy()
    GA_list_ = find_GA_v1(weight_numpy, pwr, args)
    return GA_list_

def find_GA_v1(weight, pw, args):   
    if weight.shape[2] == 3:
        num_pattern = GA_channels_v1(pw, weight, args)
        num_pattern = num_pattern.astype(int)
    return num_pattern.tolist()

def input_ch_sorting_v1(arr, nofpatterns, X, args):
    # arr_ = np.abs(arr)
    # arr_ = weight_quantization(arr, args.wb) 
    uniques, counts = np.unique(np.abs(arr), return_counts=True)
    # print("mean", np.mean(np.abs(arr)))
    # print(np.unique(arr_),np.unique(arr_).shape)
    if args.percentage == 0:
        th = 0
    else:
        th = np.percentile(np.abs(uniques), args.percentage) 
    # print("th:", th)
    arr_ = np.where(np.abs(arr) <= th, 0, arr)
    # print(np.unique(arr_),np.unique(arr_).shape)
    imp = np.sum(np.abs(arr_), (0,2,3))
    # print(imp)
    imp_sorting = np.sort(imp)
    nofprune = imp_sorting[:np.sum(nofpatterns)].tolist() # 여기서 슬라이싱을 통해 실제 적용할 부분만 셀렉팅

    imp_ = imp.tolist()
    channel_numbering = []
    for i in nofprune:
        channel_numbering.append(imp_.index(i))

    return channel_numbering




def GA_channels_v1(PW, param, args):

    skipp = 0
    ic = param.shape[1]
    best = 0

    if args.model == "WRN16-4_Q":
        if ic == 64:
            skipp = args.skipp[0]
        elif ic == 128:
            skipp = args.skipp[1]
        elif ic == 256:
            skipp = args.skipp[2]  

    def f(X):
        if not ((np.sum(X) <= ic) and ((4*PW-4)*X[0]+(3*PW-2)*X[1]+(2*PW-1)*X[2]+PW*X[3]+X[4] == skipp)):
            importance_rate = 0
            rate = 1e-10

        else: 
            nonlocal best
            new_arr = np.zeros(param.shape)
            new_X = X.astype(int)
            total_importance = np.sum(np.abs(param))
            nofentry, nofpatterns = np.nonzero(new_X)[0], new_X[np.nonzero(new_X)]
            channel_numbering = input_ch_sorting_v1(param, nofpatterns, new_X, args)

            channel_pattern_entry = []
            for i in range(len(nofpatterns)):
                for _ in range(nofpatterns[i]):
                    channel_pattern_entry.append(nofentry[i])
        
            for i in range(param.shape[1]):
                if i in channel_numbering:
                    tmp_pattern = []
                    pattern_ = channel_pattern_entry[channel_numbering.index(i)]
                    pattern_set = args.pattern_set["pattern_" + str(pattern_)]

                    for patt in pattern_set :
                        patt = np.array(patt).reshape(3,3)
                        patt = np.tile(patt, reps = [param.shape[0],1,1,1]) 
                        patt = patt.reshape(-1,9)
                        tmp_pattern.append(patt)
                    
                    pat_set = np.array(tmp_pattern)
                    pat_set = pat_set.reshape(len(pat_set), -1)
                    pat_arr = param[:,i].reshape(-1) * pat_set
                    pat_arr = np.linalg.norm(pat_arr, axis=1)
                    pat_idx = np.argmax(pat_arr)
                    new_arr[:,i] = (param[:,i].reshape(-1) * pat_set[pat_idx]).reshape(param[:,0].shape)


                else:
                    pattern_set = args.pattern_set["pattern_5"]
                    tmp_pattern = []
            
                    for patt in pattern_set :
                        patt = np.array(patt).reshape(3,3)
                        patt = np.tile(patt, reps = [param.shape[0],1,1,1]) 
                        patt = patt.reshape(-1,9)
                        tmp_pattern.append(patt)

                    pat_set = np.array(tmp_pattern)
                    pat_set = pat_set.reshape(len(pat_set), -1)
                    pat_arr = param[:,i].reshape(-1) * pat_set
                    pat_arr = np.linalg.norm(pat_arr, axis=1)
                    pat_idx = np.argmax(pat_arr)
                    new_arr[:,i] = (param[:,i].reshape(-1) * pat_set[pat_idx]).reshape(param[:,0].shape)

            after_importance = np.sum(np.abs(new_arr))
            importance_rate = after_importance / total_importance 
            if importance_rate > best:
                best = importance_rate
                rows = counting(new_arr, new_arr.shape, PW, PW)
                print()
                print("skipped_rows :", rows)
                print("best :",best)
                print("output list :", X)
                rate = 1
            else:
                rate = 1e-10

        
        return -np.sum(X)*rate
    

    varbound=np.array([[0,ic*0.5]]*5)

    algorithm_param = {'max_num_iteration': 400,\
                    'population_size':200,\
                    'mutation_probability':0.1,\
                    'elit_ratio': 0.01,\
                    'crossover_probability': 0.5,\
                    'parents_portion': 0.3,\
                    'crossover_type':'uniform',\
                    'max_iteration_without_improv':None}


    model=ga(function=f,dimension=5,variable_type='int',variable_boundaries=varbound, algorithm_parameters=algorithm_param)

    model.run() 

    if ic-np.sum(model.result['variable']) <= 0:
        resi = 0
    else :
        resi = ic-np.sum(model.result['variable'])
    
    output = np.append(model.result['variable'], resi) # no pruning을 추가해주기 위해


    return output

'''

class Conv2d_Q_mask(Conv2d_Q): ## original
    def __init__(self, w_bit, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False):
        super(Conv2d_Q_mask, self).__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.w_bit = w_bit
        self.quantize_fn = weight_quantize_fn(self.w_bit)
        self.stride = stride
        

    def forward(self, X):
        weight_q = self.quantize_fn(self.weight) 
        if self.stride == 1 :
            pwr = X.size[2]
            pattern = GA_selection_v1(weight_q, pwr, args)

        weight_q = weight_q * pattern
    
        return F.conv2d(X, weight_q, self.bias, self.stride, self.padding, self.dilation, self.groups)















class Linear_Q(nn.Linear):
    def __init__(self, *args, **kwargs):
        super(Linear_Q, self).__init__(*args, **kwargs)

class Linear_Q_(Linear_Q): ## 만든거
    def __init__(self, w_bit, in_features, out_features, bias=True):
        super(Linear_Q_, self).__init__(in_features, out_features, bias=bias)
        self.w_bit = w_bit
        self.quantize_fn = weight_quantize_fn(self.w_bit)

    def forward(self, input, order=None):
        weight_q = self.quantize_fn(self.weight)
        return F.linear(input, weight_q, self.bias)




