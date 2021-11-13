import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import math
import numpy as np
import math
import os
import argparse
import shutil
import torch.optim as optim
from torch.autograd import Variable
from dataset import *
import time
from QLayer import *

__all__ = ['ResNet', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 'resnet1202']

def _weights_init(m):
    classname = m.__class__.__name__
    #print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='B', bitW=32, bitA=32, bitO=32):
        super(BasicBlock, self).__init__()
        #self.conv1 = Conv2d_R2Q(in_planes, planes, kernel_size=3, stride=stride, padding=1, bitW=bitW, bitO=bitO, bias=False)
        self.conv1 = Conv2d_R2Q_cim(in_planes, planes, kernel_size=3, stride=stride, padding=1, bitW=bitW, bitO=bitO, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.act1 = Act_Q(bitA=bitA)
        
        #self.conv2 = Conv2d_R2Q(planes, planes, kernel_size=3, stride=1, padding=1, bitW=bitW, bitO=bitO, bias=False)
        self.conv2 = Conv2d_R2Q_cim(planes, planes, kernel_size=3, stride=1, padding=1, bitW=bitW, bitO=bitO, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                    #Conv2d_R2Q(in_planes, self.expansion * planes, kernel_size=1,
                    # stride=stride, bitW=bitW, bitO=bitO, bias=False),
                    Conv2d_R2Q_cim(in_planes, self.expansion * planes, kernel_size=1,
                     stride=stride, bitW=bitW, bitO=bitO, bias=False),
                    nn.BatchNorm2d(self.expansion * planes),
                    )
         
        self.out_act = Act_Q(bitA=bitA)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += self.shortcut(x)

        out = self.out_act(out)
        
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, bitW=32, bitA=32, bitO=32, num_classes=10):
        super(ResNet, self).__init__()
        self.bitW = bitW
        self.bitA = bitA
        self.bitO = bitO
        self.in_planes = 16

        self.conv1 = Conv2d_R2Q(3, 16, kernel_size=3, stride=1, padding=1, bitW=bitW, bitO=bitO, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.act1 = Act_Q(bitA=bitA)

        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        
        self.fc = Linear_Q(64* block.expansion, num_classes, bitW=bitW)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, bitW=self.bitW, bitA=self.bitA, bitO=self.bitO))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
                
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        
        out = self.fc(out)

        return out


def resnet20(bitW=32, bitA=32, bitO=32, num_classes=10):
    return ResNet(BasicBlock, [3, 3, 3], bitW, bitA, bitO, num_classes)

My_Model = resnet20

def train(model, optimizer, criterion, train_loader, device):
    model.train()
    avg_loss = 0.
    avg_accu = 0.

    iters = len(train_loader)
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
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=1, T_mult=2)                   ##
        scheduler.step(epoch + idx / iters)


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
    parser.add_argument('--epochs', type=int, default=300, metavar='N',
                        help='number of epochs to train (default: 300)')
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
    #parser.add_argument('--QLmode', default='ReRam', type=str,
    #                    help='quantization level mode. [\'Normal\',\'ReRam\'')

    args = parser.parse_args()
        
    # Set GPU
    device = torch.device("cuda:"+args.gpu if torch.cuda.is_available() else "cpu")
    
    # Chekpoint save path
    DIR =  './logs/resnet20_test_' + args.dataset + '_w' + str(args.w) + 'a' + str(args.a) + '_sa' + str(args.sa) + '_sub_c' + args.sub_c

    if not os.path.exists(DIR):
        os.makedirs(DIR)
    with open(DIR+'/args.txt', 'w') as f:
        f.write(str(args))
    history_score = np.zeros((args.epochs + 1, 4))
    
    # Load Mnist data
    if args.dataset == 'cifar10':
        train_loader, test_loader = get_cifar10(args.batch_size, args.test_batch_size)
        num_classes = 10
    elif args.dataset == 'cifar100':
        train_loader, test_loader = get_cifar100(args.batch_size, args.test_batch_size)
        num_classes = 100
    else:
        print('Wrong Dataset')
    
    # Load Model
    '''
    if args.pretrained:
        if args.sub_c == 'v':
            model = My_Model(bitW=args.w, bitA=args.a, bitO=args.sa, sub_channel='v', pretrained=True, classes=num_class)
        else:
            model = My_Model(bitW=args.w, bitA=args.a, bitO=args.sa, sub_channel=int(args.sub_c), pretrained=True, classes=num_class)
    else:
        if args.sub_c == 'v':
            model = My_Model(bitW=args.w, bitA=args.a, bitO=args.sa, sub_channel='v', classes=num_class)
        else:
            model = My_Model(bitW=args.w, bitA=args.a, bitO=args.sa, sub_channel=int(args.sub_c), classes=num_class)
    '''
    model = My_Model(bitW=args.w, bitA=args.a, bitO=32, num_classes=num_classes)

    if args.resume:
        ckpt_path = args.resume

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
            if (epoch in [args.epochs*0.3, args.epochs*0.6 ,args.epochs*0.9]) and (args.opt=='SGD'):
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
            if test_accu == best_accu:
                save_checkpoint(model, is_best=True, filepath=DIR)
            else:
                save_checkpoint(model, is_best=False, filepath=DIR)

        print("Best accuracy: ", best_accu)
        history_score[-1][0] = best_accu
        history_score[-1][1] = args.lr
        history_score[-1][2] = args.weight_decay
        #history_score[-1][3] = args.penalty
        np.savetxt(os.path.join(DIR, 'record.txt'), history_score, fmt = '%10.5f', delimiter=',')