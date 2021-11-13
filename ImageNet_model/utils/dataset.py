import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms
import getpass


def get_mnist(batch=256, test_batch=256, download=True, data_path='/home/' + getpass.getuser() + '/torch_data/' + 'MNIST'):
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

def get_cifar10(batch=256, test_batch=256,download=True, data_path='/home/' + getpass.getuser() + '/torch_data/' + 'CIFAR10'):
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

def get_cifar100(batch=256, test_batch=256, download=True, data_path='/home/' + getpass.getuser() + '/torch_data/' + 'CIFAR100'):
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