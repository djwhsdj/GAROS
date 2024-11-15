import math
import torch
import numpy as np
# from geneticalgorithm2 import geneticalgorithm2 as ga

def GA_selction(model, pwr, args):
    idx = -1
    idxx = 0
    GA_list = []
    for name, param in model.named_parameters():
        if name.split('.')[-1] == "weight" and len(param.shape) == 4 and 'downsample' not in name and 'conv' in name:
            idx += 1
            if args.model == 'ResNet20_Q':
                not_layer = [0, 6,12]
                if idx in not_layer :
                    continue
            elif args.model == 'WRN16-4_Q':
                not_layer = [0,4,8]
                if idx in not_layer :
                    continue
            weight_numpy = param.detach().cpu().numpy()
            GA_list_ = find_GA(weight_numpy, pwr[idxx], args)
            GA_list.append(GA_list_)
            idxx += 1

    return GA_list

def find_GA(arr, pw, args):    # input arr : (d, ch, 3, 3) or (d, ch, 1, 1)   pattern_set : (6~8, 9)
    if arr.shape[2] == 3:
        cpy_arr = arr.copy()
        num_pattern = GA_channels(pw, cpy_arr.shape, args)
        num_pattern = num_pattern.astype(int)

    return num_pattern.tolist()

def GA_channels(PW, arr_size, args):
    global skipp, IC
    skipp = 0
    ic = arr_size[1]
    IC = arr_size[1]

    # total = arr_size[1]*arr_size[2]*arr_size[3]

    if args.model == "WRN16-4_Q":
        if ic == 64:
            skipp = args.skipp[0]
        elif ic == 128:
            skipp = args.skipp[1]
        elif ic == 256:
            skipp = args.skipp[2]  

    def f(X):
        pen= 0
        non_zero = 1*X[0]+2*X[1]+4*X[2]+6*X[3]+8*X[4]
        nonzero_ratio = 1
        # total = arr_size[1]*arr_size[2]*arr_size[3]

        if not ((np.sum(X) <= IC) and ((4*PW-4)*X[0]+(3*PW-2)*X[1]+(2*PW-1)*X[2]+PW*X[3]+X[4] == skipp)):
            pen = abs(IC-np.sum(X)*abs(skipp-((4*PW-4)*X[0]+(3*PW-2)*X[1]+(2*PW-1)*X[2]+PW*X[3]+X[4])))*1000
            total = arr_size[2]*arr_size[3]*np.sum(X)
            nonzero_ratio = (non_zero/total)

        return -(np.sum(X)-pen)*abs(nonzero_ratio)
    

    varbound=np.array([[0,ic]]*5)

    model=ga(function=f,dimension=5,variable_type='int',variable_boundaries=varbound)


    model.run() 
    if ic-np.sum(model.output_dict['variable']) <= 0:
        resi = 0
    else :
        resi = ic-np.sum(model.output_dict['variable'])
    model.output_dict['variable'] = np.append(model.output_dict['variable'], resi) # no pruning을 추가해주기 위해
    print(model.output_dict['variable'])
    return model.output_dict['variable']


def delete_tensor(a, del_row, device, axis):
    n = a.cpu().detach().numpy()
    n = np.delete(n, del_row, axis)
    n = torch.from_numpy(n).to(device)
    return n

def total_sum (a,b,c):
    return a+b+c
    
def input_ch_sorting(arr, nofpatterns):

    imp = np.sum(arr, (0,2,3))
    # index_ = importance.argsort() # 정렬된 어레이의 인덱스
    imp_sorting = np.sort(imp)
    nofprune = imp_sorting[:np.sum(nofpatterns)].tolist()

    imp_ = imp.tolist()
    channel_numbering = []
    for i in nofprune:
        channel_numbering.append(imp_.index(i))

    return channel_numbering

def out_ch_sorting(arr, nofpatterns):
    arr = arr.detach().cpu().numpy()
    imp = np.sum(arr, (1,2,3))
    # index_ = importance.argsort() # 정렬된 어레이의 인덱스
    imp_sorting = np.sort(imp)
    nofprune = imp_sorting[:np.sum(nofpatterns)].tolist()

    imp_ = imp.tolist()
    channel_numbering = []
    for i in nofprune:
        channel_numbering.append(imp_.index(i))


    return channel_numbering


def ratio_calculator(a, b): # 얼마나 줄어들었는지
    return (a-b)*100/a

def avg_skipped_rows(skipped_rows, n) :
    return int(skipped_rows/n)

def cycle_calculation(ic_p, oc_p, ic, oc, k, pw, skipped_rows, ar, ac):
    # original 
    original_used_rows = ic*(pw**2)
    total_oc = ((pw-k+1)**2)*oc
    ori_AR_cycle = math.ceil(original_used_rows/ar)
    ori_AC_cycle = math.ceil(total_oc/ac)

    # ori_total_cycle = ori_AR_cycle*ori_AC_cycle

    # after prunibg
    pruning_used_rows = ic_p*(pw**2)
    duplicate_total_oc = ((pw-k+1)**2)*oc_p
    used_rows = pruning_used_rows - skipped_rows

    AR_cycle = math.ceil(used_rows/ar)
    AC_cycle = math.ceil(duplicate_total_oc/ac)

    return ori_AR_cycle, ori_AC_cycle, AR_cycle, AC_cycle

    print("Compared to No pruning")
    # print('AR    cycle reduction: %s / %s : %.1f '%(AR_cycle, ori_AR_cycle, ratio_calculator(ori_AR_cycle, AR_cycle)))
    # print('AC    cycle reduction: %s / %s : %.1f '%(AC_cycle, ori_AC_cycle, ratio_calculator(ori_AC_cycle, AC_cycle)))

def cycle_calculation_p(ic, oc, k, pw, skipped_rows, args):
    if args.model == 'ResNet20_Q':
        array = [32, 64, 128]
    else:
        array = [128, 256, 512]
    AR = [0, 0, 0]
    AC = [0, 0, 0]
    AR_ori = [0, 0, 0]
    AC_ori = [0, 0, 0]

    i=-1
    for ar in array: # ar == ac
        i += 1
        original_used_rows = ic*(pw**2)
        total_oc = ((pw-k+1)**2)*oc
        AR_ori[i] = math.ceil(original_used_rows/ar)
        AC_ori[i] = math.ceil(total_oc/ar)

        used_rows = original_used_rows - skipped_rows

        AR[i] = math.ceil(used_rows/ar)
        AC[i] = (math.ceil(total_oc/ar))

    return AR_ori, AC_ori, AR, AC

def cycle_calculation_ours(ic_p, oc_p, ic, oc, k, pw, skipped_rows, args):
    array = [args.sub_array]
    AR = [0]
    AC = [0]
    AR_ori = [0]
    AC_ori = [0]

    i=-1
    for ar in array: # ar == ac
        i += 1
        original_used_rows = ic*(pw**2)
        total_oc = ((pw-k+1)**2)*oc
        AR_ori[i] = math.ceil(original_used_rows/ar)
        AC_ori[i] = math.ceil(total_oc/ar)

        original_used_rows_ = ic_p*(pw**2)
        used_rows = original_used_rows_ - skipped_rows

        AR[i] = math.ceil(used_rows/ar)
        total_oc_ = ((pw-k+1)**2)*oc_p
        AC[i] = (math.ceil(total_oc_/ar))

    return AR_ori, AC_ori, AR, AC


def vwsdk (image_col, image_row, filter_col, filter_row, in_channel, out_channel, \
                    array_row, array_col) :

    i = 0 # initialize # overlap col
    j = 1 # overlap row

    reg_pw =[]
    reg_total_cycle = [] # initialize
    reg_overlap_row = []
    reg_overlap_col = []
    reg_row_cycle = []
    reg_col_cycle = []
    reg_ICt = []
    reg_OCt = []
    
    while True :
        try :
            i += 1
            if (i + filter_col) > image_col : 
                i = 1
                j += 1
                if j + filter_row > image_row : 
                    break

            # for parallel_window computing
            reg_N_parallel_window_row = math.ceil((image_row - (filter_row + i) + 1)/i) + 1
            reg_N_parallel_window_col = math.ceil((image_col - (filter_col + j) + 1)/j) + 1
            
            # for cycle computing
            # Tiled IC
            if in_channel == 3 :
                ICt = math.floor(array_row /((filter_row + i - 1)*(filter_col + j - 1)))
                if ICt > in_channel :
                    ICt = 3
                row_cycle = math.ceil(in_channel / ICt)
            else :
                ICt = math.floor(array_row /((filter_row + i - 1)*(filter_col + j - 1)))
                row_cycle = math.ceil(in_channel / ICt)
            
            # Tiled OC
            OCt =  math.floor(array_col / (i * j))
            col_cycle = math.ceil(out_channel / OCt)
    
            reg_N_of_computing_cycle = reg_N_parallel_window_row * reg_N_parallel_window_col \
                                    * row_cycle * col_cycle
            
            if i == 1 : # initialize
                reg_pw.append(reg_N_parallel_window_row * reg_N_parallel_window_col)
                reg_total_cycle.append(reg_N_of_computing_cycle)
                reg_overlap_row.append(i)
                reg_overlap_col.append(j)
                reg_row_cycle.append(row_cycle)
                reg_col_cycle.append(col_cycle)
                reg_ICt.append(ICt)
                reg_OCt.append(OCt)

            if reg_total_cycle[0] > reg_N_of_computing_cycle :
                del reg_pw[0]
                del reg_total_cycle[0]
                del reg_overlap_row[0]
                del reg_overlap_col[0]
                del reg_row_cycle[0]
                del reg_col_cycle[0]
                del reg_ICt[0]
                del reg_OCt[0]

                reg_pw.append(reg_N_parallel_window_row * reg_N_parallel_window_col)
                reg_total_cycle.append(reg_N_of_computing_cycle)
                reg_overlap_row.append(i)
                reg_overlap_col.append(j)
                reg_row_cycle.append(row_cycle)
                reg_col_cycle.append(col_cycle)
                reg_ICt.append(ICt)
                reg_OCt.append(OCt)

    
        except ZeroDivisionError :
            continue

    return reg_pw[0], reg_total_cycle[0], reg_ICt[0], reg_OCt[0], filter_row - 1 + reg_overlap_row[0], filter_col - 1 + reg_overlap_col[0]


def SDK (image_col, image_row, filter_col, filter_row, in_channel, out_channel, array_row, array_col) :
    
    row_vector = filter_row * filter_col * in_channel
    col_vector = out_channel
    
    used_row = math.ceil(row_vector/array_row)
    used_col = math.ceil(col_vector/array_col)
    
    new_array_row = array_row * used_row
    new_array_col = array_col * used_col

    # initialize
    cycle = []
    w = []
    w.append(filter_row*filter_col)
    cycle.append(used_row*used_col*(image_row-filter_row+1)*(image_col-filter_col+1))
    
    i=0
    while True :
        i += 1
        pw_row = filter_row + i - 1 
        pw_col = filter_col + i - 1
        pw = pw_row * pw_col
        if pw*in_channel <= new_array_row and i * i * out_channel <= new_array_col :
            parallel_window_row = math.ceil((image_row - (filter_row + i) + 1)/i) + 1
            parallel_window_col = math.ceil((image_col - (filter_col + i) + 1)/i) + 1
            
            if parallel_window_row * parallel_window_row * used_row * used_col <= cycle[0] :
                del cycle[0]
                del w[0]
                cycle.append(parallel_window_row * parallel_window_col * used_row * used_col)
                w.append(pw)
            
        else :
            break
        
    return cycle[0], int(math.sqrt(w[0])), int(math.sqrt(w[0]))


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
        # print(cnt)
    # if mode == True :
    #     print("="*60)
    #     for iccc in range(IC) :
    #         if iccc < 3 :
    #             print(mask[0][iccc])

    return cnt
