import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms


def get_mnist(batch=256, test_batch=256, download=True, data_path='/home/yrchen/torch_data/MNIST'):
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(data_path, train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=batch, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(data_path, train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=test_batch, shuffle=True)
    
    return train_loader, test_loader

def get_cifar10(batch=256, test_batch=256,download=True, data_path='/home/kuohw/novatek/R2Q/cifar10'):
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(data_path, train=True, download=True,
                       transform=transforms.Compose([
                           transforms.Pad(4),
                           transforms.RandomCrop(32),
                           transforms.RandomHorizontalFlip(),
                           transforms.ToTensor(),
                           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                       ])),
        batch_size=batch, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(data_path, train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                       ])),
        batch_size=test_batch, shuffle=True)

    return train_loader, test_loader

def get_cifar100(batch=256, test_batch=256, download=True, data_path='/home/kuohw/novatek/R2Q/cifar100'):
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100(data_path, train=True, download=True,
                       transform=transforms.Compose([
                           transforms.Pad(4),
                           transforms.RandomCrop(32),
                           transforms.RandomHorizontalFlip(),
                           transforms.ToTensor(),
                           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                       ])),
        batch_size=batch, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100(data_path, train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                       ])),
        batch_size=test_batch, shuffle=True)

    return train_loader, test_loader

# From MF lab
def cifar_transform(is_training=True):
  if is_training:
    transform_list = [transforms.RandomHorizontalFlip(),
                      transforms.Pad(padding=4, padding_mode='reflect'),
                      transforms.RandomCrop(32, padding=0),
                      transforms.RandomRotation(10),
                      transforms.ToTensor(),
                      transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), ]
  else:
    transform_list = [transforms.ToTensor(),
                      transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), ]

  transform_list = transforms.Compose(transform_list)
  return transform_list

def get_imagenet(batch=256, test_batch=256, download=True, data_path='/work/u4416566/imagenet'):
    #train
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(data_path, train=True,
                                         transform=transforms.Compose([
                                             transforms.Resize(299),
                                             transforms.CenterCrop(299),
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                         ])),
        batch_size=batch, shuffle=True, num_workers=4)

    #val
    val_loader = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(data_path, train=True,
                                         transform=transforms.Compose([
                                             transforms.Resize(299),
                                             transforms.CenterCrop(299),
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                         ])),
        batch_size=batch, shuffle=True, num_workers=4)
    
    #test
    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(data_path, train=False,
                                         transform=transforms.Compose([
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                         ])),
        batch_size=test_batch, shuffle=True, num_workers=4)
    
    return train_loader, val_loader, test_loader