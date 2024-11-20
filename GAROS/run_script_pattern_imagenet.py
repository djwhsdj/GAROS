import os
import time

print('input the mode for training mode')
mode = int(input())



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


if mode == 1 : 
    models = ['ResNet18_Q'] # ResNet20_Q
    method = [4]
    dataset = ['imagenet'] 
    mask_list = [1,2,3,4,5] # 1,3,5
    wb = 4
    ar = 2048
    ac = 1024
    epoch = 0
    GPU = 0
    seeds = [240212, 240213, 240214]


elif mode == 2 : 
    models = ['ResNet18_Q'] # ResNet20_Q
    method = [3]
    dataset = ['imagenet'] 
    mask_list = [1,2,3,4,5] # 1,3,5
    wb = 4
    ar = 2048
    ac = 1024
    epoch = 10
    GPU = 1
    seeds = [240213]


elif mode == 3 : 
    models = ['ResNet18_Q'] # ResNet20_Q
    method = [4]
    dataset = ['imagenet'] 
    mask_list = [1,2,3,4,5] # 1,3,5
    wb = 4
    ar = 2048
    ac = 1024
    epoch = 10
    GPU = 2
    seeds = [240214, 240215]




for seed in seeds:
    for model in models:
        for dst in dataset :
            for name in method :
                for mask in mask_list :
                    if mask == 1 or mask == 3 or mask == 5:
                        numsets = 4
                    elif mask == 2:
                        numsets = 14
                    elif mask == 4:
                        numsets = 16

                    os.system('python3 main_imagenet.py' + ' --model ' + str(model) + ' --num_sets ' + str(numsets) + ' --dataset ' + str(dst) 
                    + ' --mask ' + str(mask) + ' --mode ' + str(name) + ' --epoch ' + str(epoch) + ' --wb ' + str(wb) + ' --GPU ' + str(GPU) + ' --ac ' + str(ac) + ' --ar ' + str(ar) + ' --seed ' + str(seed) )

                    time.sleep(10)




