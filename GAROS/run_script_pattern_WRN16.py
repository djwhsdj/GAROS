import os
import time

print('input the mode for training mode')
mode = int(input())

# seeds = [240211, 240216, 240220]

'''
2048x1024
512x256
'''


# resnet20 : lr = 1e-1
# wrn16-4 : lr = 1e-2

# if args.mode == 0:
#     name = 'No_pruning'
# elif args.mode == 1:
#     name = 'ours_ours' # no best update
# elif args.mode == 2:
#     name = 'PATDNN'
# elif args.mode == 3:
#     name = 'RANDOM'
# elif args.mode == 4:
#     name = 'PAIRS'
# elif args.mode == 5:
#     name = 'ours_rows'


if mode == 0 : 
    models = ['WRN16-4_Q'] # ResNet20_Q
    method = [1]
    dataset = ['cifar100'] 
    mask_list = [1,2,3,4,5] # 1,3,5
    wb = 2
    ar = 2048
    ac = 1024
    epoch = 20
    GPU = 1
    seeds = [240213, 240214]
    lrrr = [1e-1]

elif mode == 1 : 
    models = ['WRN16-4_Q'] # ResNet20_Q
    method = [4]
    dataset = ['cifar100'] 
    mask_list = [1,2,3,4,5] # 1,3,5
    wb = 2
    ar = 2048
    ac = 1024
    epoch = 0
    GPU = 1
    seeds = [240211, 240212, 240213, 240214]
    lrrr = [1e-1]


# elif mode == 2 : 
#     models = ['WRN16-4_Q'] # ResNet20_Q
#     method = [3]
#     dataset = ['cifar100'] 
#     mask_list = [1,2,3,4,5] # 1,3,5
#     wb = 2
#     ar = 2048
#     ac = 1024
#     epoch = 10
#     GPU = 2
#     seeds = [240220]




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

                        os.system('python3 main.py' + ' --model ' + str(model) + ' --num_sets ' + str(numsets) + ' --dataset ' + str(dst) + ' --lr ' + str(lrr)
                        + ' --mask ' + str(mask) + ' --mode ' + str(name) + ' --epoch ' + str(epoch) + ' --wb ' + str(wb) + ' --GPU ' + str(GPU) + ' --ac ' + str(ac) + ' --ar ' + str(ar) + ' --seed ' + str(seed) )

                        time.sleep(10)





