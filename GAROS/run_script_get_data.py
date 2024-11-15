import os
import time

print('input the mode for training mode')
mode = int(input())

# if args.mode == 0:
#     name = 'No_pruning'
# elif args.mode == 1:
#     name = 'ours_fixed' # no best update
# elif args.mode == 2:
#     name = 'PATDNN'
# elif args.mode == 3:
#     name = 'RANDOM'
# elif args.mode == 4:
#     name = 'PAIRS'
# elif args.mode == 5:
#     name = 'ours_variables'


if mode == 0 : 
    models = ['ResNet20_Q'] # ResNet20_Q
    method = [3]
    dataset = ['cifar10'] 
    mask_list = [5] # 1,3,5
    wb = 2
    ar = 512
    ac = 256
    epoch = 0
    GPU = 2
    seeds = [4]
    lrrr = [1e-1]

elif mode == 1 : 
    models = ['WRN16-4_Q'] # ResNet20_Q
    method = [5]
    dataset = ['cifar100'] 
    mask_list = [1,2,3,4,5] # 1,3,5
    wb = 2
    ar = 2048
    ac = 1024
    epoch = 0
    GPU = 3
    seeds = [1]
    lrrr = [1e-1]

elif mode == 2: 
    models = ['WRN16-4_Q'] # ResNet20_Q
    method = [1]
    dataset = ['cifar100'] 
    mask_list = [1,2,3,4,5] # 1,3,5
    wb = 2
    ar = 2048
    ac = 1024
    epoch = 0
    GPU = 2
    seeds = [1]
    lrrr = [1e-1]



for seed in seeds:
    for model in models:
        for dst in dataset :
            for lrr in lrrr:
                for name in method :
                    for mask in mask_list :
                        if mask == 1 or mask == 3 or mask == 5:
                            numsets = 4
                        elif mask == 2:
                            numsets = 14
                        elif mask == 4:
                            numsets = 16

                        os.system('python3 get_data_main.py' + ' --model ' + str(model) + ' --num_sets ' + str(numsets) + ' --dataset ' + str(dst) + ' --lr ' + str(lrr)
                        + ' --mask ' + str(mask) + ' --mode ' + str(name) + ' --epoch ' + str(epoch) + ' --wb ' + str(wb) + ' --GPU ' + str(GPU) + ' --ac ' + str(ac) + ' --ar ' + str(ar) + ' --seed ' + str(seed) )

                        time.sleep(5)

