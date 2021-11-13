import sys
sys.path.append("..")
import torch
import torch.nn as nn
from torch import Tensor
import torch.utils.model_zoo as model_zoo
import math
import json

import torch.nn.functional as Function

from LLSQ_Modules_v0.Quan_Act import ACT_Q as Act_Q
from LLSQ_Modules_v0.Linear_Q import Linear_Q
from LLSQ_Modules_v0.Conv2d_quan import QuantConv2d as Conv2d_R2Q

import json
from collections import OrderedDict
from typing import Type, Any, Callable, Union, List, Optional
'''
This is for CIFAR10 and CIFAR100
'''
__all__ = ['mobilenetv1_q', 'mobilenetv1']

def load_fake_quantized_state_dict(model, original_state_dict, key_map=None):
    original_state_dict = original_state_dict['state_dict']
    if not isinstance(key_map, OrderedDict):
        with open('{}'.format(key_map)) as rf:
            key_map = json.load(rf)
    for k, v in key_map.items():
        if 'num_batches_tracked' in k:
            continue
        if 'expand_' in k and model.state_dict()[k].shape != original_state_dict[v].shape:
            ori_weight = original_state_dict[v]
            new_weight = torch.cat((ori_weight, ori_weight * 2 ** 4), dim=1)
            model.state_dict()[k].copy_(new_weight)
        else:
            model.state_dict()[k].copy_(original_state_dict[v])

def conv_bn(inp, oup, stride, bitW, bitA):
    return nn.Sequential(
        #nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        Conv2d_R2Q(inp, oup, kernel_size=3, stride=stride, padding=1, bit=bitW, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True),
        Act_Q(bit=bitA),
    )


def conv_dw(inp, oup, stride, bitW, bitA):
    return nn.Sequential(
        #nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
        Conv2d_R2Q(inp, inp, kernel_size=3, stride=stride, padding=1, bit=bitW, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        nn.ReLU6(inplace=True),
        Act_Q(bit=bitA),

        #nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        Conv2d_R2Q(inp, oup, kernel_size=1, stride=1, padding=0, bit=bitW, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True),
        Act_Q(bit=bitA),
    )

class Block_Q(nn.Module):
    '''Depthwise conv + Pointwise conv'''
    def __init__(self, in_planes, out_planes, bitW, bitA, bitO, sub_channel, stride=1):
        super(Block_Q, self).__init__()
        self.conv1 = Conv2d_R2Q(in_planes, in_planes, kernel_size=3, stride=stride, padding=1, bit=bitW, groups=in_planes, bias=False)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU6(inplace=True)
        self.relu1 = Act_Q(bit=bitA)
        self.conv2 = Conv2d_R2Q(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bit=bitW, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU6(inplace=True)
        self.relu3 = Act_Q(bit=bitA)
        self.bitW = bitW
        self.bitA = bitA
        self.bitO = bitO
        self.sub_channel = sub_channel

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        out = self.relu3(x)
        return out


class MobileNetV1_Q(nn.Module):
    '''
    # (128,2) means conv planes=128, conv stride=2, by default conv stride=1
    cfg = [64, (128,2), 128, (256,2), 256, (512,2), 512, 512, 512, 512, 512, (1024,2), 1024]

    def __init__(self, bitW, bitA, bitO, sub_channel='v', num_classes=10):
        super(MobileNetV1_Q, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = Act_Q(bit=bitA)
        self.layers = self._make_layers(in_planes=32, bitW=bitW, bitA=bitA, bitO=bitO, sub_channel='v')
        #self.linear = nn.Linear(1024, num_classes)
        #self.linear = nn.Linear(7*7*1024, num_classes)
        self.linear = Linear_Q(in_features=1024, out_features=num_classes, bias=False, bit=bitW)
        self.bitW = bitW
        self.bitA = bitA
        self.bitO = bitO
        self.sub_channel = sub_channel
        self.num_classes = num_classes

    def _make_layers(self, in_planes, bitW, bitA, bitO, sub_channel):
        layers = []
        for x in self.cfg:
            out_planes = x if isinstance(x, int) else x[0]
            stride = 1 if isinstance(x, int) else x[1]
            layers.append(Block_Q(in_planes, out_planes, bitW, bitA, bitO, sub_channel, stride))
            in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.bn1(self.conv1(x))
        out = self.relu1(out)
        out = self.relu2(out)
        out = self.layers(out)
        out = Function.avg_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
    '''
    #def __init__(self, n_class=1000):
    def __init__(self, bitW, bitA, bitO, sub_channel='v', num_classes=10):
        super(MobileNetV1_Q, self).__init__()
        self.bitW = bitW
        self.bitA = bitA
        self.bitO = bitO
        self.sub_channel = sub_channel
        self.num_classes = num_classes

        # original
        in_planes = 32
        cfg = [64, (128, 2), 128, (256, 2), 256, (512, 2), 512, 512, 512, 512, 512, (1024, 2), 1024]        

        self.conv1 = conv_bn(3, in_planes, stride=2, bitW=bitW, bitA=bitA)

        self.features = self._make_layers(in_planes, cfg, conv_dw, bitW, bitA)

        self.classifier = nn.Sequential(
            #nn.Linear(cfg[-1], num_classes),
            Linear_Q(in_features=cfg[-1], out_features=num_classes, bias=False, bit=bitW),
        )

        #self._initialize_weights()

    def forward(self, x):
        x = self.conv1(x)
        x = self.features(x)
        x = x.mean(3).mean(2)  # global average pooling

        x = self.classifier(x)
        return x

    def _make_layers(self, in_planes, cfg, layer, bitW, bitA):
        layers = []
        for x in cfg:
            out_planes = x if isinstance(x, int) else x[0]
            stride = 1 if isinstance(x, int) else x[1]
            layers.append(layer(in_planes, out_planes, stride, bitW, bitA))
            in_planes = out_planes
        return nn.Sequential(*layers)

    #def _initialize_weights(self):
    #    for m in self.modules():
    #        if isinstance(m, nn.Conv2d):
    #            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
    #            m.weight.data.normal_(0, math.sqrt(2. / n))
    #            if m.bias is not None:
    #                m.bias.data.zero_()
    #        elif isinstance(m, nn.BatchNorm2d):
    #            m.weight.data.fill_(1)
    #            m.bias.data.zero_()
    #        elif isinstance(m, nn.Linear):
    #            n = m.weight.size(1)
    #            m.weight.data.normal_(0, 0.01)
    #            m.bias.data.zero_()

def mobilenetv1_q(pretrained: bool = False, bitW=32, bitA=32, bitO=32, sub_channel='v', num_classes=10, **kwargs):
    return MobileNetV1_Q(bitW=bitW, bitA=bitA, bitO=bitO, num_classes=num_classes, sub_channel=sub_channel, **kwargs)



class Block(nn.Module):
    '''Depthwise conv + Pointwise conv'''
    def __init__(self, in_planes, out_planes, bitW, bitA, bitO, sub_channel, stride=1):
        super(Block, self).__init__()
        #self.conv1 = Conv2d_R2Q(in_planes, in_planes, kernel_size=3, stride=stride, padding=1, bitW=bitW, bitO=bitO, sub_channel=sub_channel, groups=in_planes, bias=False)
        self.conv1 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=stride, padding=1, groups=in_planes, bias=False)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        #self.conv2 = Conv2d_R2Q(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bitW=bitW, bitO=bitO, sub_channel=sub_channel, bias=False)
        self.conv2 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        #self.relu = Act_Q(bitA=bitA)
        self.relu2 = nn.ReLU(inplace=True)
        self.bitW = bitW
        self.bitA = bitA
        self.bitO = bitO
        self.sub_channel = sub_channel

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        out = self.relu2(x)
        return out


class MobileNetV1(nn.Module):
    # (128,2) means conv planes=128, conv stride=2, by default conv stride=1
    cfg = [64, (128,2), 128, (256,2), 256, (512,2), 512, 512, 512, 512, 512, (1024,2), 1024]

    def __init__(self, bitW, bitA, bitO, sub_channel='v', num_classes=10):
        super(MobileNetV1, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU(inplace=True)
        self.layers = self._make_layers(in_planes=32, bitW=bitW, bitA=bitA, bitO=bitO, sub_channel='v')
        self.linear = nn.Linear(1024, num_classes, bias=False)
        #self.linear = nn.Linear(7*7*1024, num_classes)
        #self.linear = Linear_Q(in_features=1024, out_features=num_classes, bias=False, bitW=bitW)
        #self.relu = Act_Q(bitA=bitA)
        self.relu = nn.ReLU(inplace=True)
        self.bitW = bitW
        self.bitA = bitA
        self.bitO = bitO
        self.sub_channel = sub_channel
        self.num_classes = num_classes

    def _make_layers(self, in_planes, bitW, bitA, bitO, sub_channel):
        layers = []
        for x in self.cfg:
            out_planes = x if isinstance(x, int) else x[0]
            stride = 1 if isinstance(x, int) else x[1]
            layers.append(Block(in_planes, out_planes, bitW, bitA, bitO, sub_channel, stride))
            in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.bn1(self.conv1(x))
        out = self.relu1(out)
        out = self.layers(out)
        out = F.avg_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def mobilenetv1(pretrained: bool = False, bitW=32, bitA=32, bitO=32, sub_channel='v', num_classes=10, **kwargs):
    return MobileNetV1(bitW=bitW, bitA=bitA, bitO=bitO, num_classes=num_classes, sub_channel=sub_channel, **kwargs)



