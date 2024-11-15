import os
import time

print('input the mode for training mode')
option = int(input())


if option == 0 : 
    models = ['ResNet18_Q'] 
    dataset = ['imagenet'] 
    mask_list = [6] 
    method = [6]
    wb = 4
    ar = 2048
    ac = 1024
    epoch = 10
    GPU = 0
    seeds = [240215]
    a = [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800]


elif option == 1 : 
    models = ['ResNet18_Q']
    dataset = ['imagenet'] 
    mask_list = [6] 
    method = [6]
    wb = 4
    ar = 2048
    ac = 1024
    epoch = 10
    GPU = 1
    seeds = [240215]
    a = [2000, 2200, 2400, 2600, 2800, 3000, 3200, 3400, 3600]



for seed in seeds:
    for model in models:
        for dst in dataset :
            for mod in method :
                for mas in mask_list:
                    rows = a
                    for row in rows:
                        if mas == 1 or mas == 3 or mas == 5:
                            numsets = 4
                        elif mas == 2:
                            numsets = 14
                        elif mas == 4 or mas == 8:
                            numsets = 16
                        else:
                            numsets = 10 # just test

                        os.system('python3 main_imagenet.py' + ' --model ' + str(model) + ' --dataset ' + str(dst) + ' --num_sets ' + str(numsets) 
                        + ' --epoch ' + str(epoch) + ' --GPU ' + str(GPU) + ' --ac ' + str(ac) + ' --ar ' + str(ar) + ' --rows ' + str(row) 
                        + ' --seed ' + str(seed) + ' --mode ' + str(mod) + ' --mask ' + str(mas)
                        )

                time.sleep(10)

