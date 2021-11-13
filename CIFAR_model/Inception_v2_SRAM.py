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
from QLayer_PACT_YR import *
#-------------------------------------
import time

class BasicConv2d(nn.Module):

    def __init__(self, input_channels, output_channels, bitW, bitA, bitO, **kwargs):
        super().__init__()
        self.conv = Conv2d_SRAM(input_channels, output_channels, bitW=bitW, bitA=bitA, bitO=bitO, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(output_channels)
        self.relu = Act_Q(bitA=bitA)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x

#same naive inception module
class InceptionA(nn.Module):

    def __init__(self, input_channels, pool_features, bitW, bitA, bitO):
        super().__init__()
        self.branch1x1 = BasicConv2d(input_channels, 64, bitW, bitA, bitO, kernel_size=1)

        self.branch5x5 = nn.Sequential(
            BasicConv2d(input_channels, 48, bitW, bitA, bitO, kernel_size=1),
            BasicConv2d(48, 64, bitW, bitA, bitO, kernel_size=5, padding=2)
        )

        self.branch3x3 = nn.Sequential(
            BasicConv2d(input_channels, 64, bitW, bitA, bitO, kernel_size=1),
            BasicConv2d(64, 96, bitW, bitA, bitO, kernel_size=3, padding=1),
            BasicConv2d(96, 96, bitW, bitA, bitO, kernel_size=3, padding=1)
        )

        self.branchpool = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            BasicConv2d(input_channels, pool_features, bitW, bitA, bitO, kernel_size=3, padding=1)
        )
    
    def forward(self, x):
        
        #x -> 1x1(same)
        branch1x1 = self.branch1x1(x)

        #x -> 1x1 -> 5x5(same)
        branch5x5 = self.branch5x5(x)
        #branch5x5 = self.branch5x5_2(branch5x5)

        #x -> 1x1 -> 3x3 -> 3x3(same)
        branch3x3 = self.branch3x3(x)

        #x -> pool -> 1x1(same)
        branchpool = self.branchpool(x)

        outputs = [branch1x1, branch5x5, branch3x3, branchpool]

        return torch.cat(outputs, 1)

#downsample
#Factorization into smaller convolutions
class InceptionB(nn.Module):

    def __init__(self, input_channels, bitW, bitA, bitO):
        super().__init__()

        self.branch3x3 = BasicConv2d(input_channels, 384, bitW, bitA, bitO, kernel_size=3, stride=2)

        self.branch3x3stack = nn.Sequential(
            BasicConv2d(input_channels, 64, bitW, bitA, bitO, kernel_size=1),
            BasicConv2d(64, 96, bitW, bitA, bitO, kernel_size=3, padding=1),
            BasicConv2d(96, 96, bitW, bitA, bitO, kernel_size=3, stride=2)
        )

        self.branchpool = nn.MaxPool2d(kernel_size=3, stride=2)

    def forward(self, x):

        #x - > 3x3(downsample)
        branch3x3 = self.branch3x3(x)

        #x -> 3x3 -> 3x3(downsample)
        branch3x3stack = self.branch3x3stack(x)

        #x -> avgpool(downsample)
        branchpool = self.branchpool(x)

        #"""We can use two parallel stride 2 blocks: P and C. P is a pooling 
        #layer (either average or maximum pooling) the activation, both of 
        #them are stride 2 the filter banks of which are concatenated as in 
        #figure 10."""
        outputs = [branch3x3, branch3x3stack, branchpool]

        return torch.cat(outputs, 1)
    
#Factorizing Convolutions with Large Filter Size
class InceptionC(nn.Module):
    def __init__(self, input_channels, channels_7x7, bitW, bitA, bitO):
        super().__init__()
        self.branch1x1 = BasicConv2d(input_channels, 192, bitW, bitA, bitO, kernel_size=1)

        c7 = channels_7x7

        #In theory, we could go even further and argue that one can replace any n × n 
        #convolution by a 1 × n convolution followed by a n × 1 convolution and the 
        #computational cost saving increases dramatically as n grows (see figure 6).
        self.branch7x7 = nn.Sequential(
            BasicConv2d(input_channels, c7, bitW, bitA, bitO, kernel_size=1),
            BasicConv2d(c7, c7, bitW, bitA, bitO, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(c7, 192, bitW, bitA, bitO, kernel_size=(1, 7), padding=(0, 3))
        )

        self.branch7x7stack = nn.Sequential(
            BasicConv2d(input_channels, c7, bitW, bitA, bitO, kernel_size=1),
            BasicConv2d(c7, c7, bitW, bitA, bitO, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(c7, c7, bitW, bitA, bitO, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(c7, c7, bitW, bitA, bitO, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(c7, 192, bitW, bitA, bitO, kernel_size=(1, 7), padding=(0, 3))
        )

        self.branch_pool = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            BasicConv2d(input_channels, 192, bitW, bitA, bitO, kernel_size=1),
        )

    def forward(self, x):

        #x -> 1x1(same)
        branch1x1 = self.branch1x1(x)

        #x -> 1layer 1*7 and 7*1 (same)
        branch7x7 = self.branch7x7(x)

        #x-> 2layer 1*7 and 7*1(same)
        branch7x7stack = self.branch7x7stack(x)

        #x-> avgpool (same)
        branchpool = self.branch_pool(x)

        outputs = [branch1x1, branch7x7, branch7x7stack, branchpool]

        return torch.cat(outputs, 1)

class InceptionD(nn.Module):

    def __init__(self, input_channels, bitW, bitA, bitO):
        super().__init__()

        self.branch3x3 = nn.Sequential(
            BasicConv2d(input_channels, 192, bitW, bitA, bitO, kernel_size=1),
            BasicConv2d(192, 320, bitW, bitA, bitO, kernel_size=3, stride=2)
        )

        self.branch7x7 = nn.Sequential(
            BasicConv2d(input_channels, 192, bitW, bitA, bitO, kernel_size=1),
            BasicConv2d(192, 192, bitW, bitA, bitO, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(192, 192, bitW, bitA, bitO, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(192, 192, bitW, bitA, bitO, kernel_size=3, stride=2)
        )

        self.branchpool = nn.AvgPool2d(kernel_size=3, stride=2)
    
    def forward(self, x):

        #x -> 1x1 -> 3x3(downsample)
        branch3x3 = self.branch3x3(x)

        #x -> 1x1 -> 1x7 -> 7x1 -> 3x3 (downsample)
        branch7x7 = self.branch7x7(x)

        #x -> avgpool (downsample)
        branchpool = self.branchpool(x)

        outputs = [branch3x3, branch7x7, branchpool]

        return torch.cat(outputs, 1)
    

#same
class InceptionE(nn.Module):
    def __init__(self, input_channels, bitW, bitA, bitO):
        super().__init__()
        self.branch1x1 = BasicConv2d(input_channels, 320, bitW, bitA, bitO, kernel_size=1)

        self.branch3x3_1 = BasicConv2d(input_channels, 384, bitW, bitA, bitO, kernel_size=1)
        self.branch3x3_2a = BasicConv2d(384, 384, bitW, bitA, bitO, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3_2b = BasicConv2d(384, 384, bitW, bitA, bitO, kernel_size=(3, 1), padding=(1, 0))
            
        self.branch3x3stack_1 = BasicConv2d(input_channels, 448, bitW, bitA, bitO, kernel_size=1)
        self.branch3x3stack_2 = BasicConv2d(448, 384, bitW, bitA, bitO, kernel_size=3, padding=1)
        self.branch3x3stack_3a = BasicConv2d(384, 384, bitW, bitA, bitO, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3stack_3b = BasicConv2d(384, 384, bitW, bitA, bitO, kernel_size=(3, 1), padding=(1, 0))

        self.branch_pool = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            BasicConv2d(input_channels, 192, bitW, bitA, bitO, kernel_size=1)
        )

    def forward(self, x):

        #x -> 1x1 (same)
        branch1x1 = self.branch1x1(x)

        # x -> 1x1 -> 3x1
        # x -> 1x1 -> 1x3
        # concatenate(3x1, 1x3)
        #"""7. Inception modules with expanded the filter bank outputs. 
        #This architecture is used on the coarsest (8 × 8) grids to promote 
        #high dimensional representations, as suggested by principle 
        #2 of Section 2."""
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [
            self.branch3x3_2a(branch3x3),
            self.branch3x3_2b(branch3x3)
        ]
        branch3x3 = torch.cat(branch3x3, 1)

        # x -> 1x1 -> 3x3 -> 1x3
        # x -> 1x1 -> 3x3 -> 3x1
        #concatenate(1x3, 3x1)
        branch3x3stack = self.branch3x3stack_1(x)
        branch3x3stack = self.branch3x3stack_2(branch3x3stack)
        branch3x3stack = [
            self.branch3x3stack_3a(branch3x3stack),
            self.branch3x3stack_3b(branch3x3stack)
        ]
        branch3x3stack = torch.cat(branch3x3stack, 1)

        branchpool = self.branch_pool(x)

        outputs = [branch1x1, branch3x3, branch3x3stack, branchpool]

        return torch.cat(outputs, 1)

class my_model(nn.Module):
    
    def __init__(self, bitW, bitA, bitO, sub_channel, num_classes=10):
        super().__init__()
        self.Conv2d_1a_3x3 = BasicConv2d(3, 32,  bitW, bitA, bitO, kernel_size=3, padding=1)
        self.Conv2d_2a_3x3 = BasicConv2d(32, 32,  bitW, bitA, bitO, kernel_size=3, padding=1)
        self.Conv2d_2b_3x3 = BasicConv2d(32, 64,  bitW, bitA, bitO, kernel_size=3, padding=1)
        self.Conv2d_3b_1x1 = BasicConv2d(64, 80,  bitW, bitA, bitO, kernel_size=1)
        self.Conv2d_4a_3x3 = BasicConv2d(80, 192,  bitW, bitA, bitO, kernel_size=3)

        #naive inception module
        self.Mixed_5b = InceptionA(192, pool_features=32, bitW=bitW, bitA=bitA, bitO=bitO)
        self.Mixed_5c = InceptionA(256, pool_features=64, bitW=bitW, bitA=bitA, bitO=bitO)
        self.Mixed_5d = InceptionA(288, pool_features=64, bitW=bitW, bitA=bitA, bitO=bitO)

        #downsample
        self.Mixed_6a = InceptionB(288, bitW, bitA, bitO)

        self.Mixed_6b = InceptionC(768, channels_7x7=128, bitW=bitW, bitA=bitA, bitO=bitO)
        self.Mixed_6c = InceptionC(768, channels_7x7=160, bitW=bitW, bitA=bitA, bitO=bitO)
        self.Mixed_6d = InceptionC(768, channels_7x7=160, bitW=bitW, bitA=bitA, bitO=bitO)
        self.Mixed_6e = InceptionC(768, channels_7x7=192, bitW=bitW, bitA=bitA, bitO=bitO)

        #if aux_logits:
        #    self.AuxLogits = inception_aux(768, num_classes)
        #downsample
        self.Mixed_7a = InceptionD(768, bitW=bitW, bitA=bitA, bitO=bitO)

        self.Mixed_7b = InceptionE(1280, bitW=bitW, bitA=bitA, bitO=bitO)
        self.Mixed_7c = InceptionE(2048, bitW=bitW, bitA=bitA, bitO=bitO)
        
        #6*6 feature size
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout2d()
        self.linear = nn.Linear(2048, num_classes)

    def forward(self, x):

        #32 -> 30
        x = self.Conv2d_1a_3x3(x)
        x = self.Conv2d_2a_3x3(x)
        x = self.Conv2d_2b_3x3(x)
        x = self.Conv2d_3b_1x1(x)
        x = self.Conv2d_4a_3x3(x)

        #30 -> 30
        x = self.Mixed_5b(x)
        x = self.Mixed_5c(x)
        x = self.Mixed_5d(x)

        #30 -> 14
        #Efficient Grid Size Reduction to avoid representation
        #bottleneck
        x = self.Mixed_6a(x)

        #14 -> 14
        #"""In practice, we have found that employing this factorization does not 
        #work well on early layers, but it gives very good results on medium 
        #grid-sizes (On m × m feature maps, where m ranges between 12 and 20). 
        #On that level, very good results can be achieved by using 1 × 7 convolutions 
        #followed by 7 × 1 convolutions."""
        x = self.Mixed_6b(x)
        x = self.Mixed_6c(x)
        x = self.Mixed_6d(x)
        x = self.Mixed_6e(x)

        #14 -> 6
        #Efficient Grid Size Reduction
        x = self.Mixed_7a(x)

        #6 -> 6
        #We are using this solution only on the coarsest grid, 
        #since that is the place where producing high dimensional 
        #sparse representation is the most critical as the ratio of 
        #local processing (by 1 × 1 convolutions) is increased compared 
        #to the spatial aggregation."""
        x = self.Mixed_7b(x)
        x = self.Mixed_7c(x)

        #6 -> 1
        x = self.avgpool(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x

def My_Model(bitW, bitA, bitO, sub_channel='v', pretrained=False, classes=10):
    model = my_model(bitW=bitW, bitA=bitA, bitO=bitO, sub_channel=sub_channel, classes=classes)
    model_pretrained_dict = torch.load('/home/kuohw/novatek/R2Q/CIFAR_model/logs/inceptionv2_cifar10_w8a8_sa32_sub_cv/checkpoint.pth.tar')          #['state_dict']
    #model_pretrained_dict = torch.load('/home/kuohw/novatek/R2Q/CIFAR_model/logs/inceptionv2_cifar10_w4a4_sa32_sub_cv/checkpoint.pth.tar')          #['state_dict']
    
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
    parser.add_argument('--epochs', type=int, default=300, metavar='N',
                        help='number of epochs to train (default: 300)')
    parser.add_argument('--opt', type=str, default='Adam',
                        help='SGD or Adam (default: Adam)')
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
    DIR =  './logs/inceptionv2_sram_' + args.dataset + '_w' + str(args.w) + 'a' + str(args.a) + '_sa' + str(args.sa) + '_sub_c' + args.sub_c

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
        model = My_Model(bitW=args.w, bitA=args.a, bitO=args.sa, sub_channel='v')
    else:
        model = My_Model(bitW=args.w, bitA=args.a, bitO=args.sa, sub_channel=int(args.sub_c))

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