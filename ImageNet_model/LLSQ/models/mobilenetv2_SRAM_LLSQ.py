from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
import json
import torch
from collections import OrderedDict

import math
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.model_zoo as model_zoo

from LLSQ_Modules.Conv2d_quan_mobile import QuantConv2d as Conv2d_quan
from LLSQ_Modules.Conv2d_quan_mobile import Conv2d_CIM_SRAM as Conv2d_quan_cim
from LLSQ_Modules.Conv2d_quan_mobile import Conv2d_CIM_SRAM_dw as Conv2d_quan_cim_dw
from LLSQ_Modules.Conv2d_quan_mobile import RoundFn_LLSQ, RoundFn_Bias
from LLSQ_Modules.Quan_Act_mobile import RoundFn_act, ACT_Q
from LLSQ_Modules.Linear_Q import Linear_Q

__all__ = ['MobileNetV2Q', 'mobilenetv2_q']


model_urls = {
    'mobilenet_v2': 'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth',
}

def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

class ConvBNReLU_first_layer(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1, ReLU6 = True, nbits_w=32, nbits_a=32):
        padding = (kernel_size - 1) // 2
        if ReLU6:
            super(ConvBNReLU_first_layer, self).__init__(
                Conv2d_quan(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False, bit=nbits_w),
                nn.BatchNorm2d(out_planes),
                nn.ReLU6(inplace=True),            
                ACT_Q(bit=nbits_a)
            )
        else:
            super(ConvBNReLU, self).__init__(
                Conv2d_quan(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False, bit=nbits_w),
                nn.BatchNorm2d(out_planes)
            )


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1, ReLU6 = True, nbits_w=32, nbits_a=32, adc_list=None):
        padding = (kernel_size - 1) // 2
        if ReLU6:
            if groups == 1:
                super(ConvBNReLU, self).__init__(
                    #Conv2d_quan(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False, bit=nbits_w),
                    Conv2d_quan_cim(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False, bit=nbits_w, adc_list=adc_list),
                    nn.BatchNorm2d(out_planes),
                    nn.ReLU6(inplace=True),            
                    ACT_Q(bit=nbits_a)
                )
            else:
                # dw
                super(ConvBNReLU, self).__init__(
                    #Conv2d_quan(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False, bit=nbits_w),
                    Conv2d_quan_cim_dw(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False, bit=nbits_w, adc_list=None),
                    nn.BatchNorm2d(out_planes),
                    nn.ReLU6(inplace=True),            
                    ACT_Q(bit=nbits_a)
                )
        else:
            super(ConvBNReLU, self).__init__(
                #Conv2d_quan(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False, bit=nbits_w),
                Conv2d_quan_cim(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False, bit=nbits_w, adc_list=adc_list),
                nn.BatchNorm2d(out_planes)
            )


class InvertedResidualQ(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, first, nbits_w=32, nbits_a=32, adc_list=None):
        super(InvertedResidualQ, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        if first:
            layers = []
        else:
            layers = [ACT_Q(bit=nbits_a, signed=True)]

        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1, nbits_w=nbits_w, nbits_a=nbits_a, adc_list=adc_list))
        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim, nbits_w=nbits_w, nbits_a=nbits_a, adc_list=adc_list),
            # pw-linear
            #Conv2d_quan(hidden_dim, oup, 1, 1, 0, bias=False, bit=nbits_w),
            Conv2d_quan_cim(hidden_dim, oup, 1, 1, 0, bias=False, bit=nbits_w, adc_list=adc_list),
            nn.BatchNorm2d(oup),
        ])
        
        self.conv = nn.Sequential(*layers)
        self.out_actq = ACT_Q(bit=nbits_a, signed=True)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.out_actq( self.conv(x) )['act']
        else:
            return self.conv(x)


class MobileNetV2Q(nn.Module):
    def __init__(self,
                 nbits_w=32,
                 nbits_a=32,
                 num_classes=1000,
                 width_mult=1.0,
                 inverted_residual_setting=None,
                 round_nearest=8,
                 block=None,
                 adc_list=None):
        """
        MobileNet V2 main class
        Args:
            num_classes (int): Number of classes
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            inverted_residual_setting: Network structure
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
            block: Module specifying inverted residual building block for mobilenet
        """
        super(MobileNetV2Q, self).__init__()
        
        self.nbits_w = nbits_w
        self.nbits_a = nbits_a
        self.adc_list = adc_list

        if block is None:
            block = InvertedResidualQ
        input_channel = 32
        last_channel = 1280

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 4-element list, got {}".format(inverted_residual_setting))

        # building first layer
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        features = [ConvBNReLU_first_layer(3, input_channel, stride=2, nbits_w=self.nbits_w, nbits_a=self.nbits_a)]

        # building inverted residual blocks
        first_block = True
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t, first=first_block, nbits_w=self.nbits_w, nbits_a=self.nbits_a, adc_list=self.adc_list))
                input_channel = output_channel
                first_block = False

        # building last several layers
        features.append(ACT_Q(bit=self.nbits_a, signed=True))
        features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1, ReLU6=False, nbits_w=self.nbits_w, nbits_a=self.nbits_a, adc_list=self.adc_list))
        # make it nn.Sequential
        self.features = nn.Sequential(*features)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            Linear_Q(self.last_channel, num_classes, bias=True, bit=nbits_w),
        )

        #self.relu6 = ops.ReLU6(False)
        self.relu = nn.ReLU(inplace=True)
        self.act = ACT_Q(bit=self.nbits_a)

        # weight initialization
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

    def _forward(self, x):
        x = self.features(x)
        x = self.relu(x)
        x = self.act(x)['act']
        x = x.mean([2, 3])
        #x = self.act(x)
        x = self.classifier(x)
        return x

    # Allow for accessing forward method in a inherited class
    forward = _forward

def load_fake_quantized_state_dict(model, checkpoint, key_map=None):
    if not isinstance(key_map, OrderedDict):
        with open('/home/u4416566/R2Q/01_train_from_scratch/PROFIT/pretrained/{}'.format(key_map)) as rf:
            key_map = json.load(rf)
    for k, v in key_map.items():
        if 'num_batches_tracked' in k:
            continue
        if 'expand_' in k and model.state_dict()[k].shape != checkpoint[v].shape:
            ori_weight = checkpoint[v]
            new_weight = torch.cat((ori_weight, ori_weight * 2 ** 4), dim=1)
            model.state_dict()[k].copy_(new_weight)
        else:
            model.state_dict()[k].copy_(checkpoint[k])

def mobilenetv2_q(nbits_w=32, nbits_a=32, pretrained=False, progress=True, adc_list=None, **kwargs):
    """
    Constructs a MobileNetV2 architecture from
    `"MobileNetV2: Inverted Residuals and Linear Bottlenecks" <https://arxiv.org/abs/1801.04381>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = MobileNetV2Q(nbits_w=nbits_w, nbits_a=nbits_a, adc_list=adc_list, **kwargs)
    if pretrained:
        #load_fake_quantized_state_dict(model, model_zoo.load_url(model_urls['mobilenetv2'], map_location='cpu', progress=progress),
        #                               'mobilenetv2_q_map.json')
        load_fake_quantized_state_dict(model, torch.load("./checkpoint/mobilenetv2_llsq_add1b_9a8w/ts_mobilenetv2_llsq_resnet101_llsq_ema_9_8_bn2_best.pth"),
                                       'mobilenetv2_q_map.json')
    return model
