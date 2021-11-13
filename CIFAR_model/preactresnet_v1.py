"""preactresnet in pytorch

[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun 

    Identity Mappings in Deep Residual Networks
    https://arxiv.org/abs/1603.05027
"""
# CUDA_VISIBLE_DEVICES= python3 train.py -net preact_resnet20_cifar -w 5 -b 128 -epoch 40 -finetune -finetune_pth  -lr 0.001 -checkpoint_path -bt 1000 -cifar  
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
from LLSQ_Modules.Conv2d_quan import RoundFn_LLSQ, RoundFn_Bias
from LLSQ_Modules.Conv2d_quan import QuantConv2d as Conv2dQ
from LLSQ_Modules.Conv2d_quan import Conv2d_CIM_SRAM as Conv2dQ_cim
from LLSQ_Modules.Quan_Act import RoundFn_act
from LLSQ_Modules.Quan_Act import ACT_Q as ActQ
from LLSQ_Modules.Linear_Q import Linear_Q
NUM_CLASSES = 10
 

               
##########################################################################################################################################################
def _weights_init(m):
    classname = m.__class__.__name__
    #print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)
        
class PreactBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, block_gates, inplanes, planes, stride=1, downsample=None, preact_downsample=False, nbits_w=32, nbits_a=32):
        super(PreactBasicBlock, self).__init__()
        self.block_gates = block_gates
        self.pre_bn = nn.BatchNorm2d(inplanes)
                
        self.pre_relu = nn.ReLU(inplace=True)
        self.conv1 = Conv2dQ(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False, bit=nbits_w)  
   
        self.bn = nn.BatchNorm2d(planes)
        
 
        self.relu = nn.ReLU(inplace=True)
        self.act = ActQ(bit=nbits_a)
        self.conv2 = Conv2dQ(planes, planes, kernel_size=3, stride=1, padding=1, bias=False, bit=nbits_w)
        self.act_signed = ActQ(bit=nbits_a - 1, signed=True)     
  	
        self.downsample = downsample
        self.stride = stride
        self.preact_downsample = preact_downsample
        self.out_actq = ActQ(bit=nbits_a)

    def forward(self, x):
        #need_preact = self.block_gates[0] or self.block_gates[1] or self.downsample and self.preact_downsample
        need_preact = self.downsample and self.preact_downsample


        if need_preact:


            preact = self.pre_bn(x)

            preact = self.pre_relu(preact)


            out = preact
        else:
            preact = out = x

        if self.block_gates[0]:
            out = self.conv1(out)


            out = self.bn(out)

            out = self.relu(out)
            out = self.act(out)


        if self.block_gates[1]:
            out = self.conv2(out)
            out = self.bn(out)
            out = self.act_signed(out)


        if self.downsample is not None:
            if self.preact_downsample:
                residual = self.downsample(preact)
            else:
                residual = self.downsample(x)
        else: #downsample : None
            residual = x


        out += residual
        
        #out = self.act_signed(out)
        out = self.relu(out)         # for rue
        out = self.out_actq(out)     # for rue

        return out        
class PreactResNetCifar(nn.Module):
    def __init__(self, block, layers, nbits_w=32, nbits_a=32, num_classes=NUM_CLASSES, conv_downsample=False):
        self.nbits_w = nbits_w
        self.nbits_a = nbits_a
        self.nlayers = 0
        # Each layer manages its own gates
        self.layer_gates = []
        for layer in range(3): #0 1 2
            # For each of the 3 layers, create block gates: each block has two layers
            self.layer_gates.append([])  # [True, True] * layers[layer]) 
            for blk in range(layers[layer]): #layers = [3,3,3]
                self.layer_gates[layer].append([True, True])
        self.inplanes =  16 # 16
        super(PreactResNetCifar, self).__init__()
        #self.conv1 = Conv2dQ(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False, bit=nbits_w)
        self.conv1 = nn.Sequential(
            Conv2dQ(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False, bit=nbits_w),
            nn.BatchNorm2d(self.inplanes),
            nn.ReLU(inplace=True),
            ActQ(bit=nbits_a),)        # for rue
        self.layer1 = self._make_layer(self.layer_gates[0], block, 16, layers[0], stride=1, conv_downsample=conv_downsample)
        self.layer2 = self._make_layer(self.layer_gates[1], block, 32, layers[1], stride=2, conv_downsample=conv_downsample)
        self.layer3 = self._make_layer(self.layer_gates[2], block, 64, layers[2], stride=2, conv_downsample=conv_downsample)							   					   
        self.final_bn = nn.BatchNorm2d(64 * block.expansion)##########64
        self.final_relu = nn.ReLU(inplace=True)
        self.final_act = ActQ(bit=nbits_a)
        self.avgpool= nn.AvgPool2d(8, stride=1)##########8
        self.fc = Linear_Q(64* block.expansion, num_classes)##########64
        self.dropout = nn.Dropout(0.5)
        self.apply(_weights_init)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, layer_gates, block, planes, blocks, stride=1, conv_downsample=False):
        downsample = None
        outplanes = planes * block.expansion
        if stride != 1 or self.inplanes != outplanes:
            if conv_downsample:
                downsample = nn.Sequential(
                    Conv2dQ(self.inplanes, outplanes, kernel_size=1,
                     stride=stride, bias=False, bit=self.nbits_w),
                    nn.BatchNorm2d(planes * block.expansion),
                    ActQ(bit=self.nbits_a - 1, signed=True),
                )
            else:
                # Identity downsample uses strided average pooling + padding instead of convolution
                pad_amount = int(self.inplanes / 2)
                downsample = nn.Sequential(
                    nn.AvgPool2d(2),
                    nn.ConstantPad3d((0, 0, 0, 0, pad_amount, pad_amount), 0)
                )
        #print('downsample: ',downsample)
        #print('conv_downsample: ',conv_downsample)

        layers = []

        layers.append(block(layer_gates[0], self.inplanes, planes, stride, downsample, conv_downsample, nbits_w=self.nbits_w, nbits_a=self.nbits_a))
        self.inplanes = outplanes
        for i in range(1, blocks):
            layers.append(block(layer_gates[i], self.inplanes, planes, stride=1, nbits_w=self.nbits_w, nbits_a=self.nbits_a))

        return nn.Sequential(*layers)

    def forward(self, x):
        
        x = self.conv1(x)
        x = self.dropout(x)        
        x = self.layer1(x)
        x = self.layer2(x)      
        x = self.layer3(x)	
        #x = self.final_bn(x)
        x = self.dropout(x)
        #x = self.final_relu(x)
        #x = self.final_act(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        # print(x)

        return x       
        
###################################################################################################################################        

        
def preact_resnet20_cifar10(nbits_w=32, nbits_a=32, num_classes=10):
    return PreactResNetCifar(PreactBasicBlock, [3, 3, 3], nbits_w, nbits_a, num_classes, False)        
def preact_resnet20_cifar(nbits_w=32, nbits_a=32, num_classes=100):
    return PreactResNetCifar(PreactBasicBlock, [3, 3, 3], nbits_w, nbits_a, num_classes, False)

My_Model = preact_resnet20_cifar10
#My_Model = preact_resnet20_cifar

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
    DIR =  './logs/(pre)actresnet_' + args.dataset + '_w' + str(args.w) + 'a' + str(args.a) + '_sa' + str(args.sa) + '_sub_c' + args.sub_c

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
    model = My_Model(nbits_w=args.w, nbits_a=args.a)

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
            save_checkpoint(model, test_accu>best_accu, filepath=DIR)

        print("Best accuracy: ", best_accu)
        history_score[-1][0] = best_accu
        history_score[-1][1] = args.lr
        history_score[-1][2] = args.weight_decay
        #history_score[-1][3] = args.penalty
        np.savetxt(os.path.join(DIR, 'record.txt'), history_score, fmt = '%10.5f', delimiter=',')