import os
import argparse
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dataset import *
from torch.autograd import Variable
import torchvision.models as modelss
#-------------------------------------
#from QLayer import *
from QLayer_LSQ import *
#-------------------------------------
import time

Conv2d_SRAM = Conv2dLSQ
Act_Q = ActLSQ 

def conv3x3_SRAM(in_channels, out_channels, bitW, bitA, bitO, sub_channel, stride=1):
    """3x3 convolution with padding"""
    return Conv2d_SRAM(in_channels, out_channels, kernel_size=3, stride=stride,
                     padding=1, bitW=bitW, bitA=bitA, bitO=bitO, sub_channel=sub_channel, bias=False)

def conv1x1_SRAM(in_channels, out_channels, bitW, bitA, bitO, sub_channel, stride=1):
    """1x1 convolution"""
    return Conv2d_SRAM(in_channels, out_channels, kernel_size=1, stride=stride,
                     padding=0, bitW=bitW, bitA=bitA, bitO=bitO, sub_channel=sub_channel, bias=False)

class Bottleneck(nn.Module):
    
    expansion = 4

    def __init__(self, in_channels, out_channels, bitW, bitA, bitO, stride=1, downsample=None, sub_channel='v'):
        super(Bottleneck, self).__init__()
        self.bitW = bitW
        self.bitA = bitA
        self.bitO  =bitO
        self.preact = nn.Sequential(
            conv1x1_SRAM(in_channels, out_channels, bitW=bitW, bitA=bitA, bitO=bitO,
                        sub_channel='v'),
            nn.BatchNorm2d(out_channels),
            Act_Q(bitA=bitA),
            conv3x3_SRAM(out_channels, out_channels, bitW=bitW, bitA=bitA, bitO=bitO,
                        sub_channel='v', stride=stride),
            nn.BatchNorm2d(out_channels),
            Act_Q(bitA=bitA),
            conv1x1_SRAM(out_channels, out_channels*4, bitW=bitW, bitA=bitA, bitO=bitO,
                        sub_channel='v'),
            nn.BatchNorm2d(out_channels*4),            
        )
        self.downsample = downsample
        self.stride = stride
        self.relu_q = Act_Q(bitA=bitA)
        self.out_ch = out_channels

    def forward(self, inputs):
        shortcut = inputs
        x = self.preact(inputs)
        
        if self.downsample is not None:
            shortcut = self.downsample(inputs)
       
        x += shortcut
        x = self.relu_q(x)
        return x

class my_model(nn.Module):
    def __init__(self, bitW, bitA, bitO, sub_channel='v', classes=10):
        super(my_model, self).__init__()
        self.in_channels = 64
        self.bitW = bitW
        self.bitA = bitA
        self.bitO = bitO
        self.sub_channel = sub_channel

        self.preact = nn.Sequential(
            Conv2d_SRAM(3, 64, 3, stride=1, padding=1, bitW=bitW, bitA=bitA, bitO=bitO,
                       sub_channel=sub_channel),
            nn.BatchNorm2d(64),
            Act_Q(bitA=bitA),
            #nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            #group1
            self._make_group(64, 64, 3, 1),
            #group2
            self._make_group(128, 128, 4, 2),
            #group3
            self._make_group(256, 256, 6, 2),
            #group4
            self._make_group(512, 512, 3, 2),
            
            #nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.linear = nn.Linear(in_features=2048, out_features=classes)
    
    def _make_group(self, in_ch, out_ch, num_block, stride):
        downsample = None
                
        if stride != 1 or self.in_channels != out_ch * 4:
            downsample = nn.Sequential(
                conv1x1_SRAM(self.in_channels, out_ch * 4, self.bitW, self.bitA, self.bitO,
                            self.sub_channel, stride=stride),
                nn.BatchNorm2d(out_ch * 4),
            )
            
        layers = []
        layers.append(Bottleneck(self.in_channels, out_ch, self.bitW, self.bitA, self.bitO, stride, downsample))
        self.in_channels = out_ch * 4
        for i in range(1, num_block):
            layers.append(Bottleneck(self.in_channels, out_ch, self.bitW, self.bitA, self.bitO))

        return nn.Sequential(*layers)

    def forward(self, x):
        
        x = self.preact(x)
        x = F.avg_pool2d(x, 4)
        #x = torch.flatten(x, 1)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        
        return x

def My_Model(bitW, bitA, bitO, sub_channel='v', pretrained=False, classes=10):
    model = my_model(bitW=bitW, bitA=bitA, bitO=bitO, sub_channel=sub_channel, classes=classes)
    #model_pretrained_dict = torch.load('/home/kuohw/novatek/R2Q/CIFAR_model/logs/resnet50_cifar10_w8a8_sa32_sub_cv/checkpoint.pth.tar')          #['state_dict']
    model_pretrained_dict = torch.load('/home/kuohw/novatek/R2Q/CIFAR_model/logs/resnet50_cifar10_w4a4_sa32_sub_cv/checkpoint.pth.tar')          #['state_dict']
    
    if pretrained:
        try:
            from torch.hub import load_state_dict_from_url
        except ImportError:
            from torch.utils.model_zoo import load_url as load_state_dict_from_url
        state_dict = model_pretrained_dict
        model_dict = model.state_dict()
        state_dict = {k: v for k, v in state_dict.items() if
                      (k in model_dict) and (model_dict[k].shape == state_dict[k].shape)}
        model_dict.update(state_dict)
        model.load_state_dict(model_dict)
        #model.load_state_dict(state_dict)
    return model
    
def train(model, optimizer, criterion, train_loader, device):
    model.train()
    avg_loss = 0.
    avg_accu = 0.

    for idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Avg loss & accuracy
        avg_loss += loss.data
        _, predicted = torch.max(outputs.data, 1)
        avg_accu += predicted.eq(targets.data).cpu().sum().item()

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if idx % 100 == 0:
            print('[{}/{} ({:.1f}%)]\tLoss: {:.6f}\t'.format(
                idx * targets.size(0), len(train_loader.dataset),
                100. * idx / len(train_loader), loss.data))

    return avg_loss / len(train_loader), avg_accu / len(train_loader.dataset) * 100


def test(model, criterion, test_loader, device):
    model.eval()
    avg_loss = 0.
    avg_accu = 0.

    for idx, (inputs, targets) in enumerate(test_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Avg loss & accuracy
        avg_loss += loss.data
        _, predicted = torch.max(outputs.data, 1)
        avg_accu += predicted.eq(targets.data).cpu().sum().item()

    return avg_loss / len(test_loader), avg_accu / len(test_loader.dataset) * 100

def save_checkpoint(model, is_best, filepath):
    torch.save(model.state_dict(), os.path.join(filepath, 'checkpoint.pth.tar'))
    if is_best:
        shutil.copyfile(os.path.join(filepath, 'checkpoint.pth.tar'), os.path.join(filepath, 'model_best.pth.tar'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Slimming CIFAR training')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        help='training dataset (default: cifar10)')
    parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                        help='input batch size for training (default: 256)')
    parser.add_argument('--test-batch-size', type=int, default=256, metavar='N',
                        help='input batch size for testing (default: 256)')
    parser.add_argument('--epochs', type=int, default=400, metavar='N',
                        help='number of epochs to train (default: 400)')
    parser.add_argument('--opt', type=str, default='SGD',
                        help='SGD or Adam (default: SGD)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 0.1)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint folder (default: none)')
    parser.add_argument('--gpu', default=0, type=str,
                        help='number of gpu (default: 0)')
    parser.add_argument('--eval', default=False, type=bool,
                        help='evaluate or not (default: False)')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
    parser.add_argument('-w', default=32, type=int,
                        help='weight bits (default: 32)')
    parser.add_argument('-a', default=32, type=int,
                        help='activation bits (default: 32)')
    parser.add_argument('--sa', default=32, type=int,
                        help='output bits (default: 32)')
    parser.add_argument('--sub_c', default='v', type=str,
                        help='sub channel (default: v)')
    parser.add_argument('--QLmode', default='ReRam', type=str,
                        help='quantization level mode. [\'Normal\',\'ReRam\'')

    args = parser.parse_args()
        
    # Set GPU
    device = torch.device("cuda:"+args.gpu if torch.cuda.is_available() else "cpu")

    # Chekpoint save path
    DIR =  './logs/resnet50_sram_lsq_' + args.dataset + '_w' + str(args.w) + 'a' + str(args.a) + '_sa' + str(args.sa) + '_sub_c' + args.sub_c

    if not os.path.exists(DIR):
        os.makedirs(DIR)
    with open(DIR+'/args.txt', 'w') as f:
        f.write(str(args))
    history_score = np.zeros((args.epochs + 1, 4))
    
    # Load Mnist data
    if args.dataset == 'cifar10':
        train_loader, test_loader = get_cifar10(args.batch_size, args.test_batch_size)
        num_class = 10
    elif args.dataset == 'cifar100':
        train_loader, test_loader = get_cifar100(args.batch_size, args.test_batch_size)
        num_class = 100
    else:
        print('Wrong Dataset')
    
    # Load Model
    if args.sub_c == 'v':
        model = My_Model(bitW=args.w, bitA=args.a, bitO=args.sa, sub_channel='v', classes=num_class)
    else:
        model = My_Model(bitW=args.w, bitA=args.a, bitO=args.sa, sub_channel=int(args.sub_c), classes=num_class)

    if args.resume:
        path = args.resume
        ckpt_path = path + '/checkpoint.pth.tar'

        if os.path.isfile(ckpt_path):
            print("=> loading checkpoint '{}'".format(ckpt_path))
            ckpt = torch.load(ckpt_path)
            model.load_state_dict(ckpt)
        else:
            print("=> no checkpoint found at '{}'".format(ckpt_path))
        
    model = model.to(device)
        

    # Optimizer & Loss
    if args.opt == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.opt == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=args.weight_decay)
    else:
        raise('Wrong Optimizer')
    criterion = nn.CrossEntropyLoss().to(device)
        
    # Train or Evaluation
    if args.eval:
        # Evaluation
        test_loss, test_accu = test(model=model,
                                    criterion=criterion,
                                    test_loader=test_loader, 
                                    device=device)
        print('Test set: Average loss: {:.4f}, Accuracy: {:.2f}\n'.format(test_loss, test_accu))
    else:    
        # Training 
        best_accu = 0.
        for epoch in range(args.epochs):
            if (epoch in [100, 200 ,300]) and (args.opt=='SGD'):
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= 0.1

            print('Epoch: ', epoch)
            
            start_time = time.time()
            train_loss, train_accu = train(model=model,
                                    optimizer=optimizer,
                                    criterion=criterion,
                                    train_loader=train_loader, 
                                    device=device)

            test_loss, test_accu   = test(model=model,
                                        criterion=criterion,
                                        test_loader=test_loader, 
                                        device=device)
            
            print('\nUse Time for Epoch {:d}: {:.2f}'.format(epoch, time.time()-start_time))
            print('Train set: Average loss: {:.4f}, Accuracy: {:.2f}'.format(train_loss, train_accu))
            print('Test set: Average loss: {:.4f}, Accuracy: {:.2f}\n'.format(test_loss, test_accu))
            best_accu = max(best_accu, test_accu)

            history_score[epoch][0] = train_loss
            history_score[epoch][1] = train_accu
            history_score[epoch][3] = test_accu
            np.savetxt(os.path.join(DIR, 'record.txt'), history_score, fmt = '%10.5f', delimiter=',')
            save_checkpoint(model, test_accu>best_accu, filepath=DIR)

        print("Best accuracy: ", best_accu)
        history_score[-1][0] = best_accu
        history_score[-1][1] = args.lr
        history_score[-1][2] = args.weight_decay
        #history_score[-1][3] = args.penalty
        np.savetxt(os.path.join(DIR, 'record.txt'), history_score, fmt = '%10.5f', delimiter=',') 
