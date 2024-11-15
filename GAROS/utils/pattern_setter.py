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
from tqdm import tqdm

import math
# from geneticalgorithm2 import geneticalgorithm2 as ga
import torch.backends.cudnn as cudnn


def pattern_setter_gradual_pat(model, mask_list, candidate, method, num_sets=8):
    patterns = [[0,0,0,0,0,0,0,0,0,  0]]


    idxx = -1
    for name, param in model.named_parameters():
        if name.split('.')[-1] == "weight" and len(param.shape) == 4 and 'downsample' not in name and param[0,:,:,:].shape == param[:,0,:,:].shape:
            if name[:5] == 'conv1' :
                continue
            
            print(f'name:{name}')
            idxx += 1
            par=param.cpu().detach().numpy()

            par1 = np.multiply(mask_list[idxx], np.abs(par))
            par1 = np.where(par1>0, 1, 0)
            par1 = par1.reshape(-1,1,9)
            patterns = get_pattern(patterns, par1)

            # par = np.multiply(mask_list[idxx], par)
            # patterns = get_pattern(patterns, top_4(par, candidate, method))

    patterns = np.array(patterns, dtype='int')
    patterns = patterns[patterns[:,9].argsort(kind='mergesort')]
    patterns = np.flipud(patterns)

    pattern_set = patterns[:num_sets,:9]

    
    return pattern_set


def get_pattern_gradual(patterns, arr):               # input : (?, 1, 9) / output : (?, 10) 
    l = len(arr)


    for j in range(l):
        found_flag = 0
        for i in range(len(patterns)):
            # print(arr[j].tolist())
            if np.array_equal([patterns[i][0:9]], arr[j].tolist()):
                patterns[i][9] = patterns[i][9]+1
                found_flag = 1
                break
    
        if(found_flag == 0):
            y = np.c_[arr[j], [1]]
            if sum(y.tolist()[0]) != 0 :
                patterns.append(y.tolist()[0])

    return patterns


def get_pattern(patterns, arr):               # input : (?, 1, 9) / output : (?, 10) 
    l = len(arr)

    for j in range(l):
        found_flag = 0
        for i in range(len(patterns)):
            if np.array_equal([patterns[i][0:9]], arr[j].tolist()):
                patterns[i][9] = patterns[i][9]+1
                found_flag = 1
                break;

        if(found_flag == 0):
            y = np.c_[arr[j], [1]]
            patterns.append(y.tolist()[0])
    return patterns    


# patdnn
def top_4(arr, candidate):                     # input : (d, ch, 1, 9) / output : (d*ch, 1, 9)
    arr = arr.reshape(-1,1,9)
    arr = abs(arr)
    for i in range(len(arr)):
        arr[i][0][4] = 0 
        x = arr[i].copy()
        x.sort()
        arr[i]=np.where(arr[i]<x[0][candidate+1], 0, 1) 
        arr[i][0][4] = 1 
    return arr      

# 센터 고려ㅌ   
def top_4_v(arr, candidate):                     # input : (d, ch, 1, 9) / output : (d*ch, 1, 9)
    arr = arr.reshape(-1,1,9)
    arr = abs(arr)
    for i in range(len(arr)):
        x = arr[i].copy()
        x.sort()
        arr[i]=np.where(arr[i]<x[0][candidate], 0, 1) 
    return arr       
      


def pattern_setter(model, candidate, num_sets=8):
    patterns = [[0,0,0,0,0,0,0,0,0,  0]]
    
    for name, param in model.named_parameters():
        if name.split('.')[-1] == "weight" and len(param.shape) == 4 and 'downsample' not in name and param[0,:,:,:].shape == param[:,0,:,:].shape:
            print(f'name:{name}')
            par=param.cpu().detach().numpy() 
            patterns=get_pattern(patterns, top_4_v(par, candidate))
 
    patterns = np.array(patterns, dtype='int')
    patterns = patterns[patterns[:,9].argsort(kind='mergesort')]
    patterns = np.flipud(patterns)

    pattern_set = patterns[:num_sets,:9]
    # print(pattern_set)
    
    return pattern_set

def pattern_setter_v(model, candidate, num_sets=8):
    patterns = [[0,0,0,0,0,0,0,0,0,  0]]
    
    for name, param in model.named_parameters():
        if name.split('.')[-1] == "weight" and len(param.shape) == 4 and 'downsample' not in name and param[0,:,:,:].shape == param[:,0,:,:].shape:
            print(f'name:{name}')
            par=param.cpu().detach().numpy() 
            patterns=get_pattern(patterns, top_4_v(par, candidate))
 
    patterns = np.array(patterns, dtype='int')
    patterns = patterns[patterns[:,9].argsort(kind='mergesort')]
    patterns = np.flipud(patterns)

    pattern_set = patterns[:num_sets,:9]
    # print(pattern_set)
    
    return pattern_set

'''
추가된 함수
'''
def rs (pw, entry):
  if entry == 8:
    return 9-entry # 9 is kernel size
  elif entry == 6:
    return pw
#   elif entry == 5:
#     return pw+1
  elif entry == 4:
    return 2*pw-1
#   elif entry == 3:
#     return 2*pw
  elif entry == 2:
    return 3*pw-2
  elif entry == 1:
    return 4*pw-4
#   elif entry == 0:
#       return pw**2

def residual(pw, IC, args):
    x = []
    entry = [1,2,4,6,8] 
    used_rows = pw*pw*IC
    skipped_rows = 0
    # cp_skipped_rows = 0

    total_ARC = math.ceil(used_rows/args.sub_array)


    if total_ARC == 1:
        skipped_rows = 0

    elif args.cycle >= total_ARC:
        skipped_rows = 0
    
    else:
        if used_rows%args.sub_array == 0:
            skipped_rows = args.sub_array*args.cycle
        elif used_rows%args.sub_array != 0:
            # skipped_rows = used_rows - ((args.sub_array*math.floor(used_rows/args.sub_array) + (args.cycle-1)*args.sub_array))
            skipped_rows = used_rows%args.sub_array+(args.cycle-1)*args.sub_array

    # cp_skipped_rows = skipped_rows * args.cp_rate
    
    for i in entry:
        pairs = rs(pw, i)
        # if i == 0 :
        #     y = math.floor(cp_skipped_rows/pairs) # 몇개의 채널이 꺼지는지 나타내는 변수
        #     x.append(y)
        #     skipped_rows -= y*pairs
        # else :
        x.append(skipped_rows//pairs)
        skipped_rows -= pairs * (skipped_rows//pairs)

    return np.array(x)



def top_4_pat_p(arr, pattern_set, args):    # input arr : (d, ch, 3, 3) or (d, ch, 1, 1)   pattern_set : (6~8, 9)
    if arr.shape[2] == 3:
        if args.withoc == 0 : # kernel-wise
            cpy_arr = arr.reshape(-1, 9) 
            new_arr = np.zeros(cpy_arr.shape)
            pat_set = np.array(pattern_set).reshape(-1, 9)
            for i in range(len(cpy_arr)):
                pat_arr = cpy_arr[i] * pat_set
                pat_arr = np.linalg.norm(pat_arr, axis=1)
                pat_idx = np.argmax(pat_arr)

                new_arr[i] = cpy_arr[i] * pat_set[pat_idx]

            new_arr = new_arr.reshape(arr.shape)

        elif args.withoc == 1 : # array-wise
            cpy_arr = arr.copy()
            new_arr = np.zeros(cpy_arr.shape)
            tmp_pattern = []
    
            for patt in pattern_set :
                patt = np.array(patt).reshape(3,3)
                patt = np.tile(patt, reps = [arr.shape[0],1,1,1]) 
                patt = patt.reshape(-1,9)
                tmp_pattern.append(patt)
        
            pat_set = np.array(tmp_pattern)

            for i in range(len(cpy_arr[1])):
                pat_set = pat_set.reshape(len(pat_set), -1)
                pat_arr = cpy_arr[:,i].reshape(-1) * pat_set
                pat_arr = np.linalg.norm(pat_arr, axis=1)
                pat_idx = np.argmax(pat_arr)
                new_arr[:,i] = (cpy_arr[:,i].reshape(-1) * pat_set[pat_idx]).reshape(cpy_arr[:,0].shape)

                '''
                cpy_arr = arr[0].reshape(-1, 9) 
                new_arr = np.zeros(cpy_arr.shape)
                pat_set = np.array(pattern_set).reshape(-1, 9)
                for i in range(len(cpy_arr)):
                    pat_arr = cpy_arr[i] * pat_set
                    pat_arr = np.linalg.norm(pat_arr, axis=1)
                    pat_idx = np.argmax(pat_arr)
                    new_arr[i] = cpy_arr[i] * pat_set[pat_idx]
                '''

        else : # block-wise
            cpy_arr = arr.copy()
            new_arr = np.zeros(cpy_arr.shape)
            tmp_pattern = []
            for patt in pattern_set :
                patt = np.array(patt).reshape(3,3)
                patt = np.tile(patt, reps = [arr.shape[0],arr.shape[1],1,1]) 
                patt = patt.reshape(-1,9)
                tmp_pattern.append(patt)
        
            pat_set = np.array(tmp_pattern)

            pat_set = pat_set.reshape(len(pat_set), -1)
            pat_arr = cpy_arr.reshape(-1) * pat_set
            pat_arr = np.linalg.norm(pat_arr, axis=1)
            pat_idx = np.argmax(pat_arr)
            new_arr = (cpy_arr.reshape(-1) * pat_set[pat_idx]).reshape(arr.shape)
        return new_arr
    
    else:
        return arr
        
def qfn_v1(X, wb):
    n = float(2**wb - 1)
    out = np.round(X * n) / n
    return out


def weight_quantization(x, wb):
    E = np.mean(np.abs(x))
    weight = np.tanh(x)
    weight = weight / 2 / np.max(np.abs(weight)) + 0.5
    weight_q = 2 * qfn_v1(weight, wb) - 1
    weight_q = weight_q * E
    return weight_q


def input_ch_sorting(arr, nofpatterns, X, args):
    uniques, counts = np.unique(np.abs(arr), return_counts=True)
    if args.percentage == 0:
        th = 0
    else:
        th = np.percentile(np.abs(uniques), args.percentage) 
    arr_ = np.where(np.abs(arr) <= th, 0, arr)
    imp = np.sum(np.abs(arr_), (0,2,3))
    imp_sorting = np.sort(imp)
    nofprune = imp_sorting[:np.sum(nofpatterns)].tolist()

    imp_ = imp.tolist()
    channel_numbering = []
    indx = []
    for i in nofprune:
        channel_numbering.append(imp_.index(i))

    return channel_numbering

def input_ch_sorting_random(arr, nofpatterns, args):
    channel_count = arr.shape[1]  # 가정: arr는 (batch, channels, height, width) 형태
    total_channels_to_select = np.sum(nofpatterns)  # 총 선택할 채널 수

    # 채널 번호 랜덤하게 선택
    channel_numbering = np.random.choice(channel_count, total_channels_to_select, replace=False).tolist()

    return channel_numbering

# def input_ch_sorting_v1(arr, nofpatterns, X, args):
#     uniques, counts = np.unique(np.abs(arr), return_counts=True)
#     # if args.percentage == 0:
#     #     th = 0
#     # else:
#     #     th = np.percentile(np.abs(uniques), args.percentage) 
#     # arr_ = np.where(np.abs(arr) <= th, 0, arr)
#     imp = np.sum(np.abs(arr), (0,2,3))
#     # print(imp)
#     imp_sorting = np.sort(imp).tolist()
#     # print(imp_sorting)
#     # nofprune = imp_sorting[:np.sum(nofpatterns)].tolist() # 여기서 슬라이싱을 통해 실제 적용할 부분만 셀렉팅

#     imp_ = imp.tolist()
#     channel_numbering = []
#     indx = []

#     for i in imp_sorting:
#         if imp_.index(i) not in indx:
#             indx.append(imp_.index(i))
#             channel_numbering.append(imp_.index(i))
#             imp_[imp_.index(i)] = -10

#     return channel_numbering

def input_ch_sorting_v1(arr, reverse=False):
    # 각 채널의 중요도를 계산 (채널마다 값의 절대 합을 계산)
    imp = np.sum(np.abs(arr), axis=(0, 2, 3))

    # 중요도를 오름차순으로 정렬한 인덱스를 가져옴
    sorted_indices = np.argsort(imp)  # 작은 값부터 큰 값으로 인덱스를 정렬

    # 각 인덱스에 중요도에 따른 채널 번호 부여
    channel_numbering = np.zeros_like(sorted_indices)
    if reverse:
        # 역순으로 부여 (높은 중요도가 낮은 번호)
        for i, idx in enumerate(sorted_indices):
            channel_numbering[idx] = len(sorted_indices) - 1 - i
    else:
        # 일반 순서 (낮은 중요도가 낮은 번호)
        for i, idx in enumerate(sorted_indices):
            channel_numbering[idx] = i

    # print(imp)
    return channel_numbering.tolist()

def input_ch_sorting_v1_reverse(arr, nofpatterns, X, args):
    uniques, counts = np.unique(np.abs(arr), return_counts=True)
    # if args.percentage == 0:
    #     th = 0
    # else:
    #     th = np.percentile(np.abs(uniques), args.percentage) 
    # arr_ = np.where(np.abs(arr) <= th, 0, arr)
    imp = np.sum(np.abs(arr), (0,2,3))
    # print(imp)
    imp_sorting = np.sort(imp).tolist()
    imp_sorting = list(reversed(imp_sorting))
    # print(imp_sorting)
    # nofprune = imp_sorting[:np.sum(nofpatterns)].tolist() # 여기서 슬라이싱을 통해 실제 적용할 부분만 셀렉팅

    imp_ = imp.tolist()
    channel_numbering = []
    indx = []

    for i in imp_sorting:
        if imp_.index(i) not in indx:
            indx.append(imp_.index(i))
            channel_numbering.append(imp_.index(i))
            imp_[imp_.index(i)] = -10

    return channel_numbering

def input_ch_sorting_v2(weight, nofpatterns, X, args): # 교수님께서 말씀해주신 방법 

    weight_ = np.sum(np.abs(weight), axis = 0)
    weight_e = weight_.reshape(-1)
    
    # imp_sorting = np.sort(imp) # element 단위
    uniques, counts = np.unique(np.abs(weight_e), return_counts=True)

    if args.percentage == 0: # element 별로 threshod 
      th = 0
    else:
      th = np.percentile(uniques, args.percentage) 

    weight_e_th = np.where(weight_e <= th, 0, weight_e)
    weight_e_th = weight_e_th.reshape(weight_.shape)

    # print(weight_e_th)
    imp_channel = np.sum(weight_e_th, axis=(1,2)) # channel 단위
    # print(imp_channel)
    imp_sorting = np.sort(imp_channel)

    nofprune = imp_sorting[:np.sum(nofpatterns)].tolist() # 여기서 슬라이싱을 통해 실제 적용할 부분만 셀렉팅

    imp_ = imp_channel.tolist()
    channel_numbering = []
    indx = []
    for i in nofprune:
        if imp_.index(i) not in indx:
            indx.append(imp_.index(i))
            channel_numbering.append(imp_.index(i))
            imp_[imp_.index(i)] = -10

    return channel_numbering

def top_4_pat(arr, pattern_set_total, pw, GA_list, args):    # input arr : (d, ch, 3, 3) or (d, ch, 1, 1)   pattern_set : (6~8, 9)
    if arr.shape[2] == 3:
        cpy_arr = arr.copy()
        new_arr = np.zeros(cpy_arr.shape)

        # 패턴 찾기
        num_pattern = GA_list
        num_pattern = np.array(num_pattern)
        num_pattern = num_pattern.astype(int)
        nofentry, nofpatterns = np.nonzero(num_pattern)[0], num_pattern[np.nonzero(num_pattern)]
        if args.mode == 6:
            channel_numbering = input_ch_sorting_v1(cpy_arr, reverse=True)
            # channel_numbering = input_ch_sorting_random(cpy_arr, nofpatterns, args)

        else :
            channel_numbering = input_ch_sorting_v1(cpy_arr, reverse=False)
            # channel_numbering = input_ch_sorting_random(cpy_arr, nofpatterns, args)

        channel_pattern_entry = []
        for i in range(len(nofpatterns)):
            for _ in range(nofpatterns[i]):
                channel_pattern_entry.append(nofentry[i])
        # print(channel_pattern_entry)

        # print('-'*50)
        for i in range(cpy_arr.shape[1]):
            tmp_pattern = []
            pattern_ = channel_pattern_entry[channel_numbering[i]]
            pattern_set = pattern_set_total["pattern_" + str(pattern_)]
            for patt in pattern_set :
                patt = np.array(patt).reshape(3,3)
                patt = np.tile(patt, reps = [arr.shape[0],1,1,1]) 
                patt = patt.reshape(-1,9)
                tmp_pattern.append(patt)
        
            pat_set = np.array(tmp_pattern)
            pat_set = pat_set.reshape(len(pat_set), -1)
            pat_arr = cpy_arr[:,i,:,:].reshape(-1) * pat_set
            pat_arr = np.linalg.norm(pat_arr, axis=1)
            # print(pat_arr)
            pat_idx = np.argmax(pat_arr)
            # print(pat_arr)
            # if len(pat_arr) == 1:
            #     pat_idx = 0
            # else:
            #     pat_idx = np.random.choice(4) ###########
            # pat_idx = np.argmax(pat_arr)
            # print(pat_idx)
            # print(pat_set[pat_idx])
            new_arr[:,i] = (cpy_arr[:,i].reshape(-1) * pat_set[pat_idx]).reshape(cpy_arr[:,0].shape)


    return new_arr
    
        
        

def top_k_kernel(arr, perc):    # input (d, ch, 3, 3)
    if arr.shape[2] == 1:
        new_arr = arr.copy().reshape(-1, 1)    # (d*ch, 1)
    elif arr.shape[2] == 3:
        new_arr = arr.copy().reshape(-1, 9)    # (d*ch, 9)
    else:
        return arr

    k = math.ceil(arr.shape[0] * arr.shape[1] / perc)
    l2_arr = np.linalg.norm(new_arr, axis=1)
    threshold = l2_arr[np.argsort(-l2_arr)[k-1]]
    l2_arr = l2_arr >= threshold
    
    if arr.shape[2] == 1:
        new_arr = new_arr.reshape(-1) * l2_arr

    elif arr.shape[2] == 3:
        l2_arr = l2_arr, l2_arr, l2_arr, l2_arr, l2_arr, l2_arr, l2_arr, l2_arr, l2_arr
        l2_arr = np.transpose(np.array(l2_arr))
        new_arr = new_arr * l2_arr
   
    new_arr = new_arr.reshape(arr.shape)
    return new_arr


##### for 'main_swp.py' #####
def top_4_pat_swp(arr, pattern_set):   # input arr : (d, ch, 3, 3) or (d, ch, 1, 1)  pattern_set : (6~8, 9)
    if arr.shape[2] == 3:
        cpy_arr = arr.copy().reshape(len(arr), -1, 9)
        new_arr = np.zeros(cpy_arr.shape)
        pat_set = pattern_set.copy().reshape(-1, 9)
        pat_rst = np.zeros(len(pat_set))

        pat_arr = 0
        for i in range(len(cpy_arr)):
            for j in range(len(pat_set)):
                pat_arr = cpy_arr[i] * pat_set[j]
                pat_rst[j] = np.linalg.norm(pat_arr.reshape(-1))
        
            pat_idx = np.argmax(pat_rst)
            new_arr[i] = cpy_arr[i] * pat_set[pat_idx]

        new_arr = new_arr.reshape(arr.shape)
        return new_arr
    else:
        return arr



""" my mistake1... should use tensor / torch calculation! (for speed)

def top_4_pat(arr, pattern_set):    # input arr : (d, ch, 3, 3)   pattern_set : (6~8, 9) (9 is 3x3)
    cpy_arr = arr.copy().reshape(-1, 1, 9)
    new_arr = np.zeros(cpy_arr.shape)

    for i in range(len(cpy_arr)):
        max = -1
        for j in range(len(pattern_set)):
            pat_arr = cpy_arr[i] * pattern_set[j]
            pat_l2 = np.linalg.norm(cpy_arr[i])
            
            if pat_l2 > max:
                max = pat_l2
                new_arr[i] = pat_arr
        
    new_arr = new_arr.reshape(arr.shape)
    return new_arr


def top_k_kernel(arr, perc):    # input (d, ch, 3, 3)
    k = math.ceil(arr.shape[0] * arr.shape[1] / perc)
    new_arr = arr.copy().reshape(-1, 1, 9)    # (d*ch, 1, 9)
    l2_arr = np.zeros(len(new_arr))

    for i in range(len(new_arr)):
        l2_arr[i] = np.linalg.norm(new_arr[i]) 
        
    threshold = l2_arr[np.argsort(-l2_arr)[k-1]]    # top k-th l2-norm

    for i in range(len(new_arr)):
        new_arr[i] = new_arr[i] * (l2_arr[i] >= threshold)
    
    new_arr = new_arr.reshape(arr.shape)
    return new_arr
"""





