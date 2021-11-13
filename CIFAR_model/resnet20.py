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
import re
from torch.nn.utils import clip_grad_norm_
'''
from LLSQ_Modules.Conv2d_quan import RoundFn_LLSQ, RoundFn_Bias
from LLSQ_Modules.Conv2d_quan import QuantConv2d as Conv2dQ
from LLSQ_Modules.Conv2d_quan import Conv2d_CIM_SRAM as Conv2dQ_cim
from LLSQ_Modules.Quan_Act import RoundFn_act
from LLSQ_Modules.Quan_Act import ACT_Q as ActQ
from LLSQ_Modules.Linear_Q import Linear_Q
'''
from X_Modules.ConvAct import QConv as Conv2dQ
# from X_Modules.Conv2d_quan import Conv2dQ as Conv2dQ
from X_Modules.Quan_Act import ActQ
from X_Modules.Linear_Q import LinearQ as Linear_Q

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

    def __init__(self, in_planes, planes, stride=1, option='B', nbits_w=32, nbits_a=32, alpha_bit=32, comb_list=None):
        super(BasicBlock, self).__init__()
        self.conv1 = Conv2dQ(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False, bit=nbits_w)
        #self.conv1 = Conv2dQ_cim(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False, bit=nbits_w, comb_list=comb_list)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.act1 = ActQ(bit=nbits_a)
        
        self.conv2 = Conv2dQ(planes, planes, kernel_size=3, stride=1, padding=1, bias=False, bit=nbits_w)
        #self.conv2 = Conv2dQ_cim(planes, planes, kernel_size=3, stride=1, padding=1, bias=False, bit=nbits_w, comb_list=comb_list)
        self.bn2 = nn.BatchNorm2d(planes)
        #self.act2 = ActQ(bit=nbits_a - 1, signed=True, alpha_bit=alpha_bit)
        self.act2 = ActQ(bit=nbits_a, signed=True)

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
                    Conv2dQ(in_planes, self.expansion * planes, kernel_size=1,
                     stride=stride, bias=False, bit=nbits_w),
                    #Conv2dQ_cim(in_planes, self.expansion * planes, kernel_size=1,
                    # stride=stride, bias=False, bit=nbits_w, comb_list=comb_list),
                    nn.BatchNorm2d(self.expansion * planes),
                    #ActQ(bit=nbits_a - 1, signed=True, alpha_bit=alpha_bit),
                    #ActQ(bit=nbits_a, signed=True),
                    )
         
        #self.out_act = ActQ(bit=nbits_a, alpha_bit=alpha_bit)
        self.out_act = ActQ(bit=nbits_a)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        #out = self.act1(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        #out = self.act2(out)  #['act']              # class <'torch.Tensor'>
        
        #out += self.shortcut(x_dict)['act']      # class <'torch.Tensor'>
        out += self.shortcut(x)
        
        out = self.relu(out)
        #out = self.out_act(out)
        
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, nbits_w=32, nbits_a=32, alpha_bit=32, num_classes=10, comb_list=None):
        super(ResNet, self).__init__()
        self.nbits_w = nbits_w
        self.nbits_a = nbits_a
        self.alpha_bit = alpha_bit
        self.in_planes = 16
        self.comb_list = comb_list

        self.conv1 = Conv2dQ(3, 16, kernel_size=3, stride=1, padding=1, bias=False, bit=nbits_w)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.act1 = ActQ(bit=nbits_a)

        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        
        #self.fc = Linear_Q(64* block.expansion, num_classes)
        self.fc = nn.Linear(64* block.expansion, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, nbits_w=self.nbits_w, nbits_a=self.nbits_a, alpha_bit=self.alpha_bit, comb_list=self.comb_list))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        #act2int_ten('input', x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        #out = self.act1(out)

        #act2int('layer1.input', out)
        out = self.layer1(out)
        #act2int('layer2.input', out)
        out = self.layer2(out)
        #act2int('layer3.input', out)
        out_temp = self.layer3(out)
        out = out_temp  #['act']
        
        #act2int('pool.input', out_temp)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        
        #out_temp1 = {'act': out, 'alpha': out_temp['alpha']}
        #act2int_last('fc.input', out_temp1)
        out = self.fc(out)
        
        #out_temp2 = {'act': out, 'alpha': out_temp['alpha']}
        #act2int_last('output', out_temp2)
        return out


def resnet20(nbits_w=32, nbits_a=32, alpha_bit=32, num_classes=10, comb_list=None):
    return ResNet(BasicBlock, [3, 3, 3], nbits_w, nbits_a, alpha_bit, num_classes, comb_list)

My_Model = resnet20
'''
def act2int_ten(name, inp):
    return writeout_act(name, inp)
    
def act2int(name, inp):
    x = inp['act']
    alpha = inp['alpha']

    B, CH, H, W = x.shape
    
    # 16 channel as a Group
    GPS = CH//16                                             # GPS means GrouPS
    input_slices = x.view(B, GPS, 16, H, W)                  # [B, CH//16, 16, H, W]
    
    # Initialize the OFM
    # calculate output height and width
    OH = H
    OW = W
    output = torch.zeros((B, 16, OH, OW), device='cuda:0')                          # [B, 16, OH, OW]
    
    for gp in range(GPS):
        input_unfold = torch.nn.functional.unfold(input_slices[:, gp, :, :, :],     # [B, 16*KH*KW, OH*OW]
                                                kernel_size=(1, 1), 
                                                stride=1, 
                                                padding=0)
        input_unfold = input_unfold.transpose(1, 2)                                 # [B, OH*OW, 16*KH*KW]
        input_unfold = input_unfold.view(B, OH*OW, 1*1, 16)                         # [B, OH*OW, KH*KW, 16]
        
        for i in range(1*1):
            # 8a8w
            # FP --> Int
            x_int = torch.round(input_unfold[:, :, i, :] / alpha)                   # [B, OH*OW, 16]
            
        x_int = x_int.transpose(1, 2)                                               # [B, 16, OH*OW]

        if output.sum() == 0:
            output = torch.nn.functional.fold(x_int, (OH, OW), (1, 1))              # [B, 16, OH, OW]
        else:
            output = torch.cat(( output, torch.nn.functional.fold(x_int, (OH, OW), (1, 1)) ), dim=1)
    
    writeout_act(name, output.int())
    return 0

def act2int_last(name, inp):
    x = inp['act']
    alpha = inp['alpha']
    
    if name == 'fc.input':
        return writeout_act(name, torch.round(x / alpha).int())
    elif name == 'output':
        return writeout_act(name, torch.round(x / alpha / 0.4809).int())

def writeout_act(name, inp):
    reshape_inp = inp.reshape(-1)
    with open('./patterns/fmap/' + name + '.dat', 'w') as f:
        for data in reshape_inp:
            f.write(str(data.item())+'\n')
        f.close()
    return 0
'''
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
        #assert torch.isnan(loss).sum() == 0, print(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=1, T_mult=2)                   ##
        #scheduler.step(epoch + idx / iters)
        
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
'''
def gen_pattern(model, criterion, test_loader, device):
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
        break

    return avg_loss / len(test_loader), avg_accu / len(test_loader.dataset) * 100
'''
def save_checkpoint(model, is_best, filepath):
    torch.save(model.state_dict(), os.path.join(filepath, 'checkpoint.pth.tar'))
    if is_best:
        shutil.copyfile(os.path.join(filepath, 'checkpoint.pth.tar'), os.path.join(filepath, 'model_best.pth.tar'))

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

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
    parser.add_argument('--gen-pattern', default=False, type=bool,
                        help='evaluate or not (default: False)')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
    parser.add_argument('-w', default=32, type=int,
                        help='weight bits (default: 32)')
    parser.add_argument('-a', default=32, type=int,
                        help='activation bits (default: 32)')
    parser.add_argument('--alpha-bit', default=32, type=int,
                        help='alpha of activation bits (default: 32)')
    parser.add_argument('--sa', default=32, type=int,
                        help='output bits (default: 32)')
    parser.add_argument('--sub_c', default='v', type=str,
                        help='sub channel (default: v)')
    #parser.add_argument('--QLmode', default='ReRam', type=str,
    #                    help='quantization level mode. [\'Normal\',\'ReRam\'')
    # arguments for quantization
    parser.add_argument('--QWeightFlag', type=str2bool, default=True, help='do weight quantization')
    parser.add_argument('--QActFlag', type=str2bool, default=True, help='do activation quantization')
    parser.add_argument('--weight_levels', type=int, default=2, help='number of weight quantization levels')
    parser.add_argument('--act_levels', type=int, default=2, help='number of activation quantization levels')
    parser.add_argument('--baseline', type=str2bool, default=False, help='training with STE')
    parser.add_argument('--bkwd_scaling_factorW', type=float, default=0.0, help='scaling factor for weights')
    parser.add_argument('--bkwd_scaling_factorA', type=float, default=0.0, help='scaling factor for activations')
    parser.add_argument('--use_hessian', type=str2bool, default=True, help='update scsaling factor using Hessian trace')
    parser.add_argument('--update_every', type=int, default=10, help='update interval in terms of epochs')

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
    # comb_list = [torch.zeros(241, dtype=int, device='cuda:5'),
    #             torch.zeros(241, dtype=int, device='cuda:5'),
    #             torch.zeros(241, dtype=int, device='cuda:5'),
    #             torch.zeros(241, dtype=int, device='cuda:5'),
    #             torch.zeros(241, dtype=int, device='cuda:5'),
    #             torch.zeros(241, dtype=int, device='cuda:5'),
    #             ]
    model = My_Model(nbits_w=args.w, nbits_a=args.a, alpha_bit=args.alpha_bit, num_classes=num_classes, comb_list=None)
    '''
    def first_weig2int(weight, alpha_w):
        w_reshape = weight.reshape([weight.shape[0], -1]).transpose(0, 1)
        
        wq = (w_reshape / alpha_w).round().clamp(min =-128, max = 127)
        w_q = wq.transpose(0, 1).reshape(weight.shape)
        
        return w_q

    def weig2int(weight, alpha_w):
        int_weig = torch.zeros(weight.shape)

        w_reshape = weight.reshape([weight.shape[0], -1]).transpose(0, 1)
        
        wq = (w_reshape / alpha_w).round().clamp(min =-128, max = 127) * alpha_w
        w_q = wq.transpose(0, 1).reshape(weight.shape)

        K, CH, KH, KW = w_q.shape
        alpha = 16

        GPS = CH//alpha                                             # GPS means GrouPS
        weight_slices = w_q.view(K, GPS, alpha, KH, KW)          # [K, CH//alpha, 16, KH, KW]
        int_weig_slices = int_weig.view(K, GPS, alpha, KH, KW)

        for gp in range(GPS):
            weight_unfold = weight_slices[:, gp, :, :, :].view(K, -1).t()           # [alpha*KH*KW, K]
            int_weig_unfold = int_weig_slices[:, gp, :, :, :].view(K, -1).t()
            weight_unfold = weight_unfold.view(KH*KW, alpha, K)                     # [KH*KW, alpha, K]
            int_weig_unfold = int_weig_unfold.view(KH*KW, alpha, K)

            for i in range(KH*KW):
                # 8a8w
                # FP --> Int
                int_weig_unfold[i, :, :] = torch.round(torch.clamp(weight_unfold[i, :, :] / alpha_w, min=-128, max=127)).int()   # 128 <- 2**(8-1)

            int_weig_slices[:, gp, :, :, :] = int_weig_unfold.transpose(0, 2).view(K, alpha, KH, KW)        
        int_weig = int_weig_slices.view(K, GPS*alpha, KH, KW)
            
        return int_weig
    
    def writeout_weig(name, inp):
        reshape_inp = inp.reshape(-1)
        with open('./patterns/weight/' + name + '.dat', 'w') as f:
            for data in reshape_inp:
                f.write(str(data.item())+'\n')
            f.close()
        return 0
    '''
    if args.resume:
        ckpt_path = args.resume
        
        if os.path.isfile(ckpt_path):
            print("=> loading checkpoint '{}'".format(ckpt_path))
            ckpt = torch.load(ckpt_path, map_location=device)
            
            if args.gen_pattern:
                import collections
                order_dict_w = collections.OrderedDict()
                
                p1 = "(\S+)conv(\d+).weight"
                p2 = "(\S+)conv(\d+).alpha_w"
                s1 = "(\S+)shortcut.0.weight"
                s2 = "(\S+)shortcut.0.alpha_w"
                
                counter = 0
                for a, b in ckpt.items():
                    m1 = re.findall(p1, a)
                    m2 = re.findall(p2, a)
                    ms1 = re.findall(s1, a)
                    ms2 = re.findall(s2, a)
                    
                    if a == 'conv1.weight':
                        buffer = b
                    elif a == 'conv1.alpha_w':
                        order_dict_w[a[:-8]] = first_weig2int(buffer, b)
                    elif len(m1) or len(m2) or len(ms1) or len(ms2) != 0:
                        if counter == 0:                                   # weight
                            buffer = b
                            counter += 1
                        elif counter == 1:                                 # alpha_w
                            order_dict_w[a[:-8]] = weig2int(buffer, b)
                            counter = 0

                for k, v in order_dict_w.items():
                    writeout_weig(k, v.int())
                
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
    elif args.gen_pattern:
        # Evaluation
        _, test_loader = get_cifar100(1, 1)
        test_loss, test_accu = gen_pattern(model=model,
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
