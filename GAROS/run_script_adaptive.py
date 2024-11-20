import os
import time

print('input the mode for training mode')
option = int(input())

# seeds = [1106, 1016, 2023, 2022]

# if args.mode == 0:
#     name = 'No_pruning'
# elif args.mode == 1:
#     name = 'ours_fixed'
# elif args.mode == 2:
#     name = 'PATDNN'
# elif args.mode == 3:
#     name = 'RANDOM'
# elif args.mode == 4:
#     name = 'PAIRS'
# elif args.mode == 5:
#     name = 'ours_variables'
# elif args.mode == 6:
#     name = 'ours_variables_reverse'

# a = []
# for i in range(20):
#     a.append(50*(i+1))

b = []
for i in range(20):
    b.append(200*(i+1))

if option == 0 : 
    models = ['WRN16-4_Q']#['ResNet20_Q']
    dataset = ['cifar100'] 
    ar = 2048#512
    ac = 1024#256
    epoch = 20
    mask = [6]
    mode = [5]
    seeds = [240214] 
    GPU = 0
    # a = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900]
    b = [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400, 2600, 2800, 3000, 3200, 3400, 3600]
    lrrr = [1e-1]
    wb = 2


elif option == 1 : 
    models = ['WRN16-4_Q']#['ResNet20_Q']
    dataset = ['cifar100'] 
    ar = 2048#512
    ac = 1024#256
    epoch = 20
    mask = [6]
    mode = [6]
    seeds = [240214] 
    GPU = 1
    # a = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900]
    b= [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400, 2600, 2800, 3000, 3200, 3400, 3600]
    lrrr = [1e-1]
    wb = 2

elif option == 2 : 
    models = ['WRN16-4_Q']
    dataset = ['cifar100'] 
    ar = 2048
    ac = 1024
    epoch = 20
    mask = [1,2,3,4,5]
    mode = [1]
    seeds = [240211, 240212, 240213, 240214] 
    GPU = 3
    b = [0]
    lrrr = [1e-1]
    wb = 2
# elif option == 1 : 
#     models = ['WRN16-4_Q']
#     dataset = ['cifar100'] 
#     ar = 2048
#     ac = 1024
#     epoch = 0 # 100
#     mask = [6]
#     mode = [6]
#     seeds = [240211, 240212, 240213, 240214] 
#     GPU = 1
#     b = [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400, 2600, 2800, 3000, 3200, 3400, 3600]
#     lrrr = [1e-1]
#     wb = 2

# if option == 1 : 
#     models = ['ResNet20_Q']
#     dataset = ['cifar10'] 
#     ar = 512
#     ac = 256
#     epoch = 30
#     mask = [6]
#     mode = [6]
#     seeds = [240211]
#     GPU = 1
#     # a = [550, 600, 650, 700, 750, 800, 850, 900, 950, 1000, 1050]
#     a = [1000]

# if option == 2 : 
#     models = ['ResNet20_Q']
#     dataset = ['cifar10'] 
#     ar = 512
#     ac = 256
#     epoch = 10
#     mask = [6]
#     mode = [5]
#     seeds = [240211, 240216, 240220]
#     GPU = 2
#     a = [750, 800, 850, 900, 950, 1000, 1050]


# elif option == 1 : 
#     models = ['WRN16-4_Q']
#     dataset = ['cifar100'] 
#     ar = 2048
#     ac = 1024
#     epoch = 20 # 100
#     mask = [6]
#     mode = [6]
#     seeds = [240213, 240214]
#     GPU = 1
#     b = [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400, 2600, 2800, 3000, 3200, 3400, 3600]
#     lrrr = [1e-1]
    

# elif option == 2 : 
#     models = ['WRN16-4_Q']
#     dataset = ['cifar100'] 
#     ar = 2048
#     ac = 1024
#     epoch = 20 # 100
#     mask = [6]
#     mode = [6]
#     seeds = [240211, 240212]
#     GPU = 0
#     b = [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400, 2600, 2800, 3000, 3200, 3400, 3600]
#     lrrr = [1e-1]

# elif option == 3 : 
#     models = ['WRN16-4_Q']
#     dataset = ['cifar100'] 
#     ar = 2048
#     ac = 1024
#     epoch = 20 # 100
#     mask = [6]
#     mode = [5]
#     seeds = [240211]
#     GPU = 1
#     b = [3600]

# elif option == 2 : 
#     models = ['ResNet20_Q'] # ResNet20_Q
#     dataset = ['cifar10'] 
#     ar = 512
#     ac = 256
#     re_epoch = 100 # 100
#     mask = [6]
#     mode = [0]
#     seeds = [2020,2021]
#     GPU = 0

# elif option == 3 : 
#     models = ['ResNet20_Q']
#     dataset = ['cifar10'] 
#     ar = 512
#     ac = 256
#     re_epoch = 100 # 100
#     mask = [6]
#     mode = [0]
#     seeds = [2022, 2023]
#     GPU = 1
    


for mod in mode :
    for model in models:
        for dst in dataset :
            for seed in seeds:
                for lrr in lrrr:
                    for mas in mask:
                        if model == 'ResNet20_Q': 
                            rows = a
                        else:
                            rows = b

                        for row in rows:
                            if mas == 1 or mas == 3 or mas == 5:
                                numsets = 4
                            elif mas == 2:
                                numsets = 14
                            elif mas == 4 or mas == 8:
                                numsets = 16
                            else:
                                numsets = 10 # just test

                            os.system('python main.py' + ' --model ' + str(model) + ' --dataset ' + str(dst) + ' --num_sets ' + str(numsets) + ' --lr ' + str(lrr)
                            + ' --epoch ' + str(epoch) + ' --GPU ' + str(GPU) + ' --ac ' + str(ac) + ' --ar ' + str(ar) + ' --rows ' + str(row) + ' --wb ' + str(wb)
                            + ' --seed ' + str(seed) + ' --mode ' + str(mod) + ' --mask ' + str(mas)
                            )

                    time.sleep(10)

