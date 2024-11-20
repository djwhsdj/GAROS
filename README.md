# GAROS: Genetic Algorithm-Aided Row-Skipping for SDK-Based Convolutional Weight Mapping


  ## Please use 'run_script' to evaluate the methods. Before training, please check the mode for each method.
    * # if args.mode == 0:
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

  ## Running GAROS
  #### run_script_adaptive.py
    * This is a run script in ResNet-20 on CIFAR-10 and WRN16-4 on CIFAR-100
  #### run_script_adaptive_imagenet.py
    * This is a run script code in ResNet-18 on ImageNet

  ## Running Pattern-based pruning
  #### run_script_pattern_(ResNet20, WRN16-4, imagenet)
    * This is a run script code for pattern-based pruning methods in ResNet-20, WRN16-4, ResNet-18 repectively.
