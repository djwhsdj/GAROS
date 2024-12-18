import os
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, SVHN, MNIST
import torchvision.datasets as datasets

def data_loader(dir, dataset, batch_size, workers):
    if dataset == 'cifar10':
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        val_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        train_set = datasets.CIFAR10(root=dir, train=True, download=True, transform=train_transform)
        val_set = datasets.CIFAR10(root=dir, train=False, download=False, transform=val_transform)


    elif dataset == 'svhn' :
        train_transform = transforms.Compose([
            transforms.Resize(32),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        val_transform = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        train_set = datasets.SVHN(root=dir, split='train', download=True, transform=train_transform)
        val_set = datasets.SVHN(root=dir, split='test', download=False, transform=val_transform)

    elif dataset == 'cifar100':
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
        ])
        val_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
        ])
        train_set = datasets.CIFAR100(root=dir, train=True, download=True, transform=train_transform)
        val_set = datasets.CIFAR100(root=dir, train=False, download=False, transform=val_transform)


        
    elif dataset == 'imagenet':
        traindir = os.path.join(dir, 'train')
        valdir = os.path.join(dir, 'val')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                         std=[0.229, 0.224, 0.225])
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        val_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])
        train_set = datasets.ImageFolder(traindir, transform=train_transform)
        val_set = datasets.ImageFolder(valdir, transform=val_transform)
    elif dataset == 'mnist':
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
            ])
        val_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
            ])
        train_set = datasets.MNIST(root=dir, train=True, download=True, transform=train_transform)
        val_set = datasets.MNIST(root=dir, train=False, download=False, transform=val_transform)
           
    else:
        assert False, 'No Such Dataset'
        
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)

    return train_loader, val_loader
