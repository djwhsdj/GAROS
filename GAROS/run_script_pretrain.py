import os
import time

print('input the mode for training mode')
mode = int(input())

# seeds = [1106, 1016, 2023, 2022]
# seeds = [231106, 231016, 232023, 232022, 231992]
seeds = [1]

'''
2048x1024
512x512
'''

if mode == 0 : 
    models = ['ResNet20_Q']
    dataset = ['cifar10'] 
    epoch = 200
    ab = 32
    wb = 1
    GPU = 0

elif mode == 1 : # batch size 512
    models = ['ResNet18_Q']
    dataset = ['imagenet'] 
    epoch = 120
    ab = 32
    wb = 4
    GPU = 1

elif mode == 2 : 
    models = ['WRN16-4_Q']
    dataset = ['cifar100'] 
    epoch = 200
    ab = 32
    wb = 2
    GPU = 1

elif mode == 3 : 
    models = ['WRN16-4_Q']
    dataset = ['cifar100'] 
    epoch = 200
    ab = 4
    wb = 4
    GPU = 0

for seed in seeds:
    for model in models:
        for dst in dataset :
            os.system('python3 main_network_pretrain.py' + ' --model ' + str(model) + ' --dataset ' + str(dst) 
            + ' --epoch ' + str(epoch) + ' --GPU ' + str(GPU) + ' --seed ' + str(seed) + ' --ab ' + str(ab) + ' --wb ' + str(wb) )

            time.sleep(10)




