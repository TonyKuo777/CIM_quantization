import torch.nn as nn
import math
from torch.hub import load_state_dict_from_url
#---------------------------------------
#from utils.QLayer import *
#from utils.QLayer_QILwPACT import *
#from utils.Conv2d_quan import *
#from utils.Quan_Act import *
from utils.QLayer_PACT_SRAM import *
#---------------------------------------

#Conv2d_SRAM = QuantConv2d
#Act_Q = PACT

def conv_bn(inp, oup, stride, bitW=32, bitA=32, bitO=32, sub_channel='v'):
    return nn.Sequential(
        #nn.Conv2d(inp, oup, 3, stride, 1, bias=True),
        Conv2d_SRAM(inp, oup, kernel_size=3, stride=stride, padding=1, bias=True, bitW=bitW, bitO=bitO, sub_channel=sub_channel),
        nn.BatchNorm2d(oup),
        #nn.ReLU6(inplace=True)
        Act_Q(bitA=bitA),
    )

def conv_1x1_bn(inp, oup, bitW=32, bitA=32, bitO=32, sub_channel='v'):
    return nn.Sequential(
        #nn.Conv2d(inp, oup, 1, 1, 0, bias=True),
        Conv2d_SRAM(inp, oup, kernel_size=1, stride=1, padding=0, bias=True, bitW=bitW, bitO=bitO, sub_channel=sub_channel),
        nn.BatchNorm2d(oup),
        #nn.ReLU6(inplace=True)
        Act_Q(bitA=bitA)
    )

def make_divisible(x, divisible_by=8):
    import numpy as np
    return int(np.ceil(x * 1. / divisible_by) * divisible_by)


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, bitW=32, bitA=32, bitO=32, sub_channel='v'):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup
        
        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                #nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=True),
                Conv2d_SRAM(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=True, bitW=bitW, bitO=bitO, sub_channel=sub_channel),
                nn.BatchNorm2d(hidden_dim),
                #nn.ReLU6(inplace=True),
                Act_Q(bitA),

                # pw-linear
                #nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=True),
                Conv2d_SRAM(hidden_dim, oup, 1, 1, 0, bias=True, bitW=bitW, bitO=bitO, sub_channel=sub_channel),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                #nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=True),
                Conv2d_SRAM(inp, hidden_dim, 1, 1, 0, bias=True, bitW=bitW, bitO=bitO, sub_channel=sub_channel),
                nn.BatchNorm2d(hidden_dim),
                #nn.ReLU6(inplace=True),
                Act_Q(bitA),

                # dw
                #nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=True),
                Conv2d_SRAM(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=True, bitW=bitW, bitO=bitO, sub_channel=sub_channel),
                nn.BatchNorm2d(hidden_dim),
                #nn.ReLU6(inplace=True),
                Act_Q(bitA),

                # pw-linear
                #nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=True),
                Conv2d_SRAM(hidden_dim, oup, 1, 1, 0, bias=True, bitW=bitW, bitO=bitO, sub_channel=sub_channel),
                nn.BatchNorm2d(oup),
            )
            
    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)  

class MobileNetV2(nn.Module):
    def __init__(self, n_class=1000, input_size=224, width_mult=1., bitW=32, bitA=32, bitO=32, sub_channel='v'):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280
        interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # building first layer
        assert input_size % 32 == 0
        self.last_channel = make_divisible(last_channel * width_mult) if width_mult > 1.0 else last_channel
        self.features = [conv_bn(3, input_channel, 2, bitW=32, bitA=32, bitO=32, sub_channel='v')]
        
        # building inverted residual blocks
        for t, c, n, s in interverted_residual_setting:
            output_channel = make_divisible(c * width_mult) if t > 1 else c
            for i in range(n):
                if i == 0:
                    self.features.append(block(input_channel, output_channel, s, expand_ratio=t, bitW=bitW, bitA=bitA, bitO=bitO, sub_channel=sub_channel))
                else:
                    self.features.append(block(input_channel, output_channel, 1, expand_ratio=t, bitW=bitW, bitA=bitA, bitO=bitO, sub_channel=sub_channel))
                    
                input_channel = output_channel
        # building last several layers
        self.features.append(conv_1x1_bn(input_channel, self.last_channel, bitW, bitA, bitO, sub_channel))
        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)
        
        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            #nn.Linear(self.last_channel, n_class),
            Linear_Q(self.last_channel, n_class, bitW=bitW),
        )

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.mean(3).mean(2)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

def mobilenet_v2(bitW, bitA, bitO, sub_channel='v', pretrained=False):
    model = MobileNetV2(width_mult=1, bitW=bitW, bitA=bitA, bitO=bitO, sub_channel=sub_channel)
    
    if pretrained:
        try:
            from torch.hub import load_state_dict_from_url
        except ImportError:
            from torch.utils.model_zoo import load_url as load_state_dict_from_url
        state_dict = load_state_dict_from_url(
            'https://www.dropbox.com/s/47tyzpofuuyyv1b/mobilenetv2_1.0-f2a8633.pth.tar?dl=1', progress=True)
        #state_dict = torch.load('/home/kuohw/novatek/R2Q/01_train_from_scratch/logs/mobilev2_w4a4_sa32_sub_cv/model_best.pth.tar', map_location="cuda:4")['state_dict']
        #state_dict = torch.load('/home/kuohw/novatek/R2Q/01_train_from_scratch/logs/mobilev2_w4a4_sa32_sub_cv/model_best.pth.tar')['state_dict']
    
        model_dict = model.state_dict()
        state_dict = {k: v for k, v in state_dict.items() if
                      (k in model_dict) and (model_dict[k].shape == state_dict[k].shape)}
        model_dict.update(state_dict)
        model.load_state_dict(model_dict)
    return model