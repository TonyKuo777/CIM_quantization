import os
import argparse
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.optim as optim
from dataset import *
from torch.autograd import Variable
import torchvision.models as modelss
from QLayer import *
import time

class BaseBlock(nn.Module):
    alpha = 1

    def __init__(self, input_channel, output_channel, bitW, bitA, bitO, t = 6, downsample = False):
        """
            t:  expansion factor, t*input_channel is channel of expansion layer
            alpha:  width multiplier, to get thinner models
            rho:    resolution multiplier, to get reduced representation
        """ 
        super(BaseBlock, self).__init__()
        self.stride = 2 if downsample else 1
        self.downsample = downsample
        self.shortcut = (not downsample) and (input_channel == output_channel) 

        # apply alpha
        input_channel = int(self.alpha * input_channel)
        output_channel = int(self.alpha * output_channel)
        
        # for main path:
        c  = t * input_channel
        # 1x1   point wise conv
        #self.conv1 = Conv2d_R2Q(input_channel, c, kernel_size = 1, bitW = bitW, bias = False)
        self.conv1 = nn.Conv2d(input_channel, c, kernel_size = 1)
        self.bn1 = nn.BatchNorm2d(c)
        self.relu_1 = Act_Q(bitA=bitA)
        # 3x3   depth wise conv
        #self.conv2 = Conv2d_R2Q(c, c, kernel_size = 3, stride = self.stride, padding = 1, bitW = bitW, groups = c, bias = False)
        self.conv2 = nn.Conv2d(c, c, kernel_size = 3)
        self.bn2 = nn.BatchNorm2d(c)
        self.relu_2 = Act_Q(bitA=bitA)
        # 1x1   point wise conv
        #self.conv3 = Conv2d_R2Q(c, output_channel, kernel_size = 1, bitW = bitW, bias = False)
        self.conv3 = nn.Conv2d(c, output_channel, kernel_size = 1)
        self.bn3 = nn.BatchNorm2d(output_channel)
        #self.relu6_3 = Act6_Q(bitA=bitA)
        

    def forward(self, inputs):
        # main path
        x = self.relu_1(self.bn1(self.conv1(inputs)))
        x = self.relu_2(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        
        # shortcut path
        x = x + inputs if self.shortcut else x

        return x

'''
if __name__ == "__main__":
    from torchvision.datasets import CIFAR10
    import torchvision.transforms as transforms

    transform = transforms.Compose([
        transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = CIFAR10("~/dataset/cifar10", transform = transform)
    x = trainset[0][0].unsqueeze(0)

    print(x.shape)
    BaseBlock.alpha = 0.5
    b = BaseBlock(6, 5, downsample = True)
    y = b(x)
    print(b)
    print(y.shape, y.max(), y.min())
'''

class My_Model(nn.Module):
    def __init__(self, bitW, bitA, bitO, alpha = 1, sub_channel='v', mode='ReRam', classes=10):
        super(My_Model, self).__init__()
        
        # first conv layer 
        self.conv0 = Conv2d_R2Q(3, int(32*alpha), bitW = bitW, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.bn0 = nn.BatchNorm2d(int(32*alpha))
        self.relu_0 = Act_Q(bitA=bitA)

        # build bottlenecks
        BaseBlock.alpha = alpha
        self.bottlenecks = nn.Sequential(
            BaseBlock(32, 16, bitW = bitW, bitA = bitA, bitO = bitO, t = 1, downsample = False),
            BaseBlock(16, 24, bitW = bitW, bitA = bitA, bitO = bitO, downsample = False),
            BaseBlock(24, 24, bitW = bitW, bitA = bitA, bitO = bitO),
            BaseBlock(24, 32, bitW = bitW, bitA = bitA, bitO = bitO, downsample = False),
            BaseBlock(32, 32, bitW = bitW, bitA = bitA, bitO = bitO),
            BaseBlock(32, 32, bitW = bitW, bitA = bitA, bitO = bitO),
            BaseBlock(32, 64, bitW = bitW, bitA = bitA, bitO = bitO, downsample = True),
            BaseBlock(64, 64, bitW = bitW, bitA = bitA, bitO = bitO),
            BaseBlock(64, 64, bitW = bitW, bitA = bitA, bitO = bitO),
            BaseBlock(64, 64, bitW = bitW, bitA = bitA, bitO = bitO),
            BaseBlock(64, 96, bitW = bitW, bitA = bitA, bitO = bitO, downsample = False),
            BaseBlock(96, 96, bitW = bitW, bitA = bitA, bitO = bitO),
            BaseBlock(96, 96, bitW = bitW, bitA = bitA, bitO = bitO),
            BaseBlock(96, 160, bitW = bitW, bitA = bitA, bitO = bitO, downsample = True),
            BaseBlock(160, 160, bitW = bitW, bitA = bitA, bitO = bitO),
            BaseBlock(160, 160, bitW = bitW, bitA = bitA, bitO = bitO),
            BaseBlock(160, 320, bitW = bitW, bitA = bitA, bitO = bitO, downsample = False))

        # last conv layers and fc layer
        #self.conv1 = Conv2d_R2Q(int(320*alpha), 1280, bitW = bitW, kernel_size = 1, bias = False)
        self.conv1 = nn.Conv2d(int(320*alpha), 1280, kernel_size = 1)
        self.bn1 = nn.BatchNorm2d(1280)
        self.relu_1 = Act_Q(bitA=bitA)
        self.fc = nn.Linear(in_features = 1280, out_features = classes, bias = False)

        # weights init
        #self.weights_init()
        '''
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)
        '''
    def weights_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    
    def forward(self, inputs):

        # first conv layer
        x = self.relu_0(self.bn0(self.conv0(inputs)))
        # assert x.shape[1:] == torch.Size([32, 32, 32])

        # bottlenecks
        x = self.bottlenecks(x)
        # assert x.shape[1:] == torch.Size([320, 8, 8])

        # last conv layer
        x = self.relu_1(self.bn1(self.conv1(x)))
        # assert x.shape[1:] == torch.Size([1280,8,8])

        # global pooling and fc (in place of conv 1x1 in paper)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)

        return x

'''
if __name__ == "__main__":
    from torchvision.datasets import CIFAR10
    from count import measure_model
    import torchvision.transforms as transforms
    import numpy as np

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = CIFAR10("~/dataset/cifar10", transform = transform)
    x = trainset[0][0].unsqueeze(0)
    print(x.shape)

    net = MobileNetV2(10, alpha = 1)
    y = net(x)

    print(x.shape)
    print(y.shape)

    f, c = measure_model(net, 32, 32)
    print("model size %.4f M, ops %.4f M" %(c/1e6, f/1e6))

    # size = 1
    # for param in net.parameters():
    #     arr = np.array(param.size())
        
    #     s = 1
    #     for e in arr:
    #         s *= e
    #     size += s

    # print("all parameters %.2fM" %(size/1e6) )
'''
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
    DIR =  './logs/mobilenetv2_' + args.dataset + '_w' + str(args.w) + 'a' + str(args.a) + '_sa' + str(args.sa) + '_sub_c' + args.sub_c

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
        model = My_Model(bitW=args.w, bitA=args.a, bitO=args.sa, sub_channel='v', mode=args.QLmode, classes=num_class)
    else:
        model = My_Model(bitW=args.w, bitA=args.a, bitO=args.sa, sub_channel=int(args.sub_c), mode=args.QLmode, classes=num_class)

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