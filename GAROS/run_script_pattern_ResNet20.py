import os
import time

print('input the mode for training mode')
mode = int(input())

'''
2048x1024
512x256
'''

      # if args.mode == 0:
      #     name = 'No_pruning'  # original 
      # elif args.mode == 1:
      #     name = 'ours_fixed' # GAROS to compare with PAIRS
      # elif args.mode == 2:
      #     name = 'PATDNN'
      # elif args.mode == 3:
      #     name = 'RANDOM'
      # elif args.mode == 4:
      #     name = 'PAIRS'
      # elif args.mode == 5:
      #     name = 'ours_variables' # GAROS 
      # elif args.mode == 6:
      #     name = 'ours_variables_reverse' for GAROS_re or GAROS_ran


if mode == 0 : 
    models = ['ResNet20_Q'] # ResNet20_Q
    method = [4]
    dataset = ['cifar10'] 
    mask_list = [1,2,3,4,5] # 1,3,5
    wb = 2
    ar = 512
    ac = 256
    epoch = 0
    GPU = 0
    seeds = [240211, 240212, 240213, 240214]
    lrrr = [1e-1]

elif mode == 1 : 
    models = ['ResNet20_Q'] # ResNet20_Q
    method = [1]
    dataset = ['cifar10'] 
    mask_list = [1,2,3,4,5] # 1,3,5
    wb = 2
    ar = 512
    ac = 256
    epoch = 20
    GPU = 0
    seeds = [240212]
    lrrr = [1e-1]

elif mode == 2 : 
    models = ['ResNet20_Q'] # ResNet20_Q
    method = [1]
    dataset = ['cifar10'] 
    mask_list = [1,2,3,4,5] # 1,3,5
    wb = 1
    ar = 512
    ac = 256
    epoch = 20
    GPU = 1
    seeds = [240213]
    lrrr = [1e-1]

elif mode == 3 : 
    models = ['ResNet20_Q'] # ResNet20_Q
    method = [1]
    dataset = ['cifar10'] 
    mask_list = [1,2,3,4,5] # 1,3,5
    wb = 1
    ar = 512
    ac = 256
    epoch = 20
    GPU = 1
    seeds = [240214]
    lrrr = [1e-1]

# if mode == 2 : 
#     models = ['ResNet20_Q'] # ResNet20_Q
#     method = [3]
#     dataset = ['cifar10'] 
#     mask_list = [1,2,3,4,5] # 1,3,5
#     wb = 2
#     ar = 512
#     ac = 256
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

# for name in name_list :
#     for numsets in numsets_list :
#         for mask in mask_list :
#             for withoc in withoc_list :
#                 for wb in wb_list :
#                     os.system('python3 main_loss3_v1.py --lr 1e-3 --rho 1 --num_sets ' + str(numsets) 
#                     + ' --mask ' + str(mask) + ' --method ' + str(name) + ' --withoc ' + str(withoc) 
#                     + ' --wb ' + str(wb) + ' --gradual 1' + ' --GPU ' + str(GPU) + ' --epo ' + str(epo) + ' --ac ' + str(ac) + ' --ar ' + str(ar))
#                     time.sleep(10)



