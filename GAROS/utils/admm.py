import torch
import torch.nn.functional as F
import numpy as np
from .pattern_setter import *
from utils_1 import *

from model import *


def regularized_nll_loss(args, model, output, target):
    loss = F.cross_entropy(output, target)
    # cri = CrossEntropyLossSoft()
    # loss = cri(output, target)

    return loss

def admm_loss(args, device, model, Z, Y, U, V, output, target):
    idx = 0
    loss = F.cross_entropy(output, target)
    # cri = CrossEntropyLossSoft()
    # loss = cri(output, target)
    
    for name, param in model.named_parameters():
        if name.split('.')[-1] == "weight" and len(param.shape) == 4 and 'downsample' not in name and param[0,:,:,:].shape == param[:,0,:,:].shape:
            z = Z[idx].to(device)
            y = Y[idx].to(device)
            u = U[idx].to(device)
            v = V[idx].to(device)

            loss += args.rho * (param - z + u).norm() 

            '''
            loss += args.rho * 0.5 * (param - z + u).norm() 
            # + args.rho * 0.5 * (param - y + v).norm()
            '''
            idx += 1
    return loss

def re_admm_loss(args, device, model, Z, Y, U, V, output, target):
    loss = F.cross_entropy(output, target)
    idx = -1
    for name, param in model.named_parameters(): # name : weight, bias... # params : value list
        if name.split('.')[-1] == "weight" and len(param.shape) == 4 and 'downsample' not in name and param[0,:,:,:].shape == param[:,0,:,:].shape:
            idx += 1
            z = Z[idx].to(device)
            y = Y[idx].to(device)
            u = U[idx].to(device)
            v = V[idx].to(device)

            loss += args.rho * (param - z + u).norm() 
            # loss_list.append(loss_)
            # loss += loss_

    return loss

def initialize_Z_Y_U_V(model):
    Z = ()
    Y = ()
    U = ()
    V = ()
    for name, param in model.named_parameters():
        if name.split('.')[-1] == "weight" and len(param.shape) == 4 and 'downsample' not in name and param[0,:,:,:].shape == param[:,0,:,:].shape:
            Z += (param.detach().cpu().clone(),)
            Y += (param.detach().cpu().clone(),)
            U += (torch.zeros_like(param).cpu(),)
            V += (torch.zeros_like(param).cpu(),)
    return Z, Y, U, V




def re_initialize_Z_Y_V(model):
    Z = ()
    Y = ()
    V = ()
    for name, param in model.named_parameters():
        if name.split('.')[-1] == "weight" and len(param.shape) == 4 and 'downsample' not in name and param[0,:,:,:].shape == param[:,0,:,:].shape:
            # print(name, param.shape)
            Z += (param.detach().cpu().clone(),)
            Y += (param.detach().cpu().clone(),)
            V += (torch.zeros_like(param).cpu(),)
    return Z, Y, V

def update_X(model):
    X = ()
    for name, param in model.named_parameters():
        if name.split('.')[-1] == "weight" and len(param.shape) == 4 and 'downsample' not in name and param[0,:,:,:].shape == param[:,0,:,:].shape:
            X += (param.detach().cpu().clone(),)
    return X



def update_Z(X, U, pattern_set, pwset, GA_list, args):
    new_Z = ()
    a = 0
    b = 0

    if args.model == 'ResNet20_Q':
        not_layer = [0,6,12]

    elif args.model == 'WRN16-4_Q':
        not_layer = [0,4,8]

        
    for x, u in zip(X, U):
        z = x + u
        if a not in not_layer :
            # b += 1
            z = torch.from_numpy(top_4_pat(z.numpy(), pattern_set, pwset[b], GA_list[b], args))
            b += 1
            # print(a,b,len(pwset))
            #  b += 1
                #  print(a,b,len(pwset))
        # Select each kernel and prune -> z = torch.tensor (a, b, 3, 3) or (a, b, 1, 1)
        a += 1
        new_Z += (z,)
    return new_Z

def update_Z_P(X, U, pattern_set, pwset, args):
    new_Z = ()
    a = 0
    for x, u in zip(X, U):
        z = x + u
        if args.model == 'ResNet20_Q':
            not_layer = [0, 6,12]
            if a not in not_layer :
                z = torch.from_numpy(top_4_pat_p(z.numpy(), pattern_set, args))
        elif args.model == 'WRN16-4_Q':
            not_layer = [0,4,8]
            if a not in not_layer :
                 z = torch.from_numpy(top_4_pat_p(z.numpy(), pattern_set, args))

        a += 1
        new_Z += (z,)
    return new_Z


def update_Y(X, V, args):
    new_Y = ()
    for x, v in zip(X, V):
        y = x + v

        # Prune kernel by l2 -> y = torch.tensor (a, b, 3, 3) or (a, b, 1, 1) 
        y = torch.from_numpy(top_k_kernel(y.numpy(), args.connect_perc)) 

        new_Y += (y,)
    return new_Y


def update_U(U, X, Z):
    new_U = ()
    for u, x, z in zip(U, X, Z):
        new_u = u + x - z
        new_U += (new_u,)
        # print(new_U)
    return new_U


def update_V(V, X, Y):
    new_V = ()
    for v, x, y in zip(V, X, Y):
        new_v = v + x - y
        new_V += (new_v,)
    return new_V

def prune_weight_p(weight, device, percent, pattern_set, args):
    weight_numpy = weight
    # weight_numpy = weight.detach().cpu().numpy()

    # weight_numpy = top_k_kernel(weight_numpy, percent)
    weight_numpy = top_4_pat_p(weight_numpy, pattern_set, args)
    mask = np.where(np.abs(weight_numpy) > 0, 1, 0)

    
    # mask = torch.Tensor(weight_numpy != 0).to(device)
    return mask

def apply_prune_pat_p(args, model, device, pattern_set):
    dict_mask = {}
    idx = -1
    for name, param in model.named_parameters():
        if name.split('.')[-1] == "weight" and len(param.shape) == 4 and 'downsample' not in name and 'conv' in name:
            idx += 1
            if args.model == 'ResNet20_Q':
                not_layer = [6,12]
                if idx in not_layer :
                    continue
            elif args.model == 'WRN16-4_Q':
                not_layer = [0,4,8]
                if idx in not_layer :
                    continue

            mask = prune_weight_p(param, device, args.connect_perc, pattern_set, args)
            param.data.mul_(mask)
            dict_mask[name] = mask
    return dict_mask



def prune_weight(weight, device, percent, pattern_set, pw, GA_list, args):
    weight_numpy = weight

    weight_numpy = top_4_pat(weight_numpy, pattern_set, pw, GA_list, args)
    mask = np.where(np.abs(weight_numpy) > 0, 1, 0)
    return mask

def apply_prune_pat(args, model, device, pattern_set, pw, GA_list):
    dict_mask = {}
    idx = -1
    idxx = 0
    for name, param in model.named_parameters():
        if name.split('.')[-1] == "weight" and len(param.shape) == 4 and 'downsample' not in name and 'conv' in name:
            idx += 1
            if args.model == 'ResNet20_Q':
                not_layer = [6,12]
                if idx in not_layer :
                    continue
            elif args.model == 'WRN16-4_Q':
                not_layer = [0,4,8]
                if idx in not_layer :
                    continue
            
            mask = prune_weight(param, device, args.connect_perc, pattern_set, pw[idxx], GA_list[idxx], args)

            param.data.mul_(mask)
            dict_mask[name] = mask
            idxx += 1

    return dict_mask

def apply_prune(args, model, device, pattern_set):
    dict_mask = {}
    idx = -1
    idxx = 0
    for name, param in model.named_parameters():
        if name.split('.')[-1] == "weight" and len(param.shape) == 4 and 'downsample' not in name and 'conv' in name:
            idx += 1
            if args.model == 'ResNet20_Q':
                not_layer = [6,12]
                if idx in not_layer :
                    continue
            elif args.model == 'WRN16-4_Q':
                not_layer = [0,4,8]
                if idx in not_layer :
                    continue


            mask = prune_weight(param, device, args.connect_perc, pattern_set[idxx])
            param.data.mul_(mask)
            

            dict_mask[name] = mask
            idxx += 1
    return dict_mask

    
def print_convergence(model, X, Z):
    idx = 0
    print("\nnormalized norm of (weight - projection)")
    for name, _ in model.named_parameters():
        if name.split('.')[-1] == "weight":
            x, z = X[idx], Z[idx]
            print("({}): {:.4f}".format(name, (x-z).norm().item() / x.norm().item()))
            idx += 1


def print_prune(model):
    prune_param, total_param = 0, 0
    for name, param in model.named_parameters():
        if name.split('.')[-1] == "weight":
            print("[at weight {}]".format(name))
            print("percentage of pruned: {:.4f}%".format(100 * (abs(param) == 0).sum().item() / param.numel()))
            print("nonzero parameters after pruning: {} / {}\n".format((param != 0).sum().item(), param.numel()))
        total_param += param.numel()
        prune_param += (param != 0).sum().item()
    print("total nonzero parameters after pruning: {} / {} ({:.4f}%)".
          format(prune_param, total_param,
                 100 * (total_param - prune_param) / total_param))


##### for 'main_swp.py' #####
def update_Z_swp(X, U, pattern_set, args):
    new_Z = ()
    for x, u in zip(X, U):
        z = x + u
        
        # Select each kernel and prune -> z = torch.tensor (a, b, 3, 3)
        z = torch.from_numpy(top_4_pat_swp(z.numpy(), pattern_set))

        new_Z += (z,)
    return new_Z

def apply_prune_swp(args, model, device, pattern_set):
    # returns dictionary of non_zero_values' indices
    dict_mask = {}
    for name, param in model.named_parameters():
        if name.split('.')[-1] == "weight" and len(param.shape) == 4 and 'downsample' not in name and param[0,:,:,:].shape == param[:,0,:,:].shape:
            mask = prune_weight_swp(param, device, args.connect_perc, pattern_set)
            param.data.mul_(mask)
            # param.data = torch.Tensor(weight_pruned).to(device)
            dict_mask[name] = mask
    return dict_mask

def prune_weight_swp(weight, device, percent, pattern_set):
    # to work with admm, we calculate percentile based on all elements instead of nonzero elements.
    weight_numpy = weight.detach().cpu().numpy()

    weight_numpy = top_k_kernel(weight_numpy, percent)
    weight_numpy = top_4_pat_swp(weight_numpy, pattern_set)
    
    mask = torch.Tensor(weight_numpy != 0).to(device)
    return mask


##### for 'main_loss3.py test' #####
def admm_lossc(args, device, model, Z, Y, U, V, output, target):
    loss = F.cross_entropy(output, target)
    # cri = CrossEntropyLossSoft()
    # loss = cri(output, target)
    return loss

def admm_lossz(args, device, model, Z, Y, U, V, output, target):
    idx = 0
    loss = 0
    for name, param in model.named_parameters():
        if name.split('.')[-1] == "weight" and len(param.shape) == 4 and 'downsample' not in name and param[0,:,:,:].shape == param[:,0,:,:].shape:
            z = Z[idx].to(device)
            u = U[idx].to(device)

            loss += args.rho * 0.5 * (param - z + u).norm()
            idx += 1
    return loss

def admm_lossy(args, device, model, Z, Y, U, V, output, target):
    idx = 0
    loss = 0
    for name, param in model.named_parameters():
        if name.split('.')[-1] == "weight" and len(param.shape) == 4 and 'downsample' not in name and param[0,:,:,:].shape == param[:,0,:,:].shape:
            y = Y[idx].to(device)
            v = V[idx].to(device)

            loss += args.rho * 0.5 * (param - y + v).norm()
            idx += 1
    return loss


