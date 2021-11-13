import torch
import torch.nn as nn
#from .utils import load_state_dict_from_url
from torch.hub import load_state_dict_from_url
#------------------------------------------------------
#from utils.QLayer import *
from utils.QLayer_PACT_SRAM_ste import *
#------------------------------------------------------

__all__ = [
    'VGG', 'vgg16_bn',
    ]

model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}

class VGG(nn.Module):

    def __init__(self, bitW=32, bitA=32, bitO=32, adc_list=None, num_classes=1000, init_weights=True):
        super(VGG, self).__init__()

        self.conv_1 = Conv2d_SRAM(3, 64, kernel_size=3, padding=1, bias=False, bitW=bitW, bitO=bitO)
        self.bn_1 = nn.BatchNorm2d(64)
        self.act_1 = Act_Q(bitA)
        
        self.conv_2 = Conv2d_CIM_SRAM(64, 64, kernel_size=3, padding=1, bias=False, bitW=bitW, bitO=bitO, adc_list=adc_list)
        self.bn_2 = nn.BatchNorm2d(64)
        self.act_2 = Act_Q(bitA)
        
        self.conv_3 = Conv2d_CIM_SRAM(64, 128, kernel_size=3, padding=1, bias=False, bitW=bitW, bitO=bitO, adc_list=adc_list)
        self.bn_3 = nn.BatchNorm2d(128)
        self.act_3 = Act_Q(bitA)
        
        self.conv_4 = Conv2d_CIM_SRAM(128, 128, kernel_size=3, padding=1, bias=False, bitW=bitW, bitO=bitO, adc_list=adc_list)
        self.bn_4 = nn.BatchNorm2d(128)
        self.act_4 = Act_Q(bitA)
        
        self.conv_5 = Conv2d_CIM_SRAM(128, 256, kernel_size=3, padding=1, bias=False, bitW=bitW, bitO=bitO, adc_list=adc_list)
        self.bn_5 = nn.BatchNorm2d(256)
        self.act_5 = Act_Q(bitA)
        
        self.conv_6 = Conv2d_CIM_SRAM(256, 256, kernel_size=3, padding=1, bias=False, bitW=bitW, bitO=bitO, adc_list=adc_list)
        self.bn_6 = nn.BatchNorm2d(256)
        self.act_6 = Act_Q(bitA)
        
        self.conv_7 = Conv2d_CIM_SRAM(256, 256, kernel_size=3, padding=1, bias=False, bitW=bitW, bitO=bitO, adc_list=adc_list)
        self.bn_7 = nn.BatchNorm2d(256)
        self.act_7 = Act_Q(bitA)
        
        self.conv_8 = Conv2d_CIM_SRAM(256, 512, kernel_size=3, padding=1, bias=False, bitW=bitW, bitO=bitO, adc_list=adc_list)
        self.bn_8 = nn.BatchNorm2d(512)
        self.act_8 = Act_Q(bitA)
        
        self.conv_9 = Conv2d_CIM_SRAM(512, 512, kernel_size=3, padding=1, bias=False, bitW=bitW, bitO=bitO, adc_list=adc_list)
        self.bn_9 = nn.BatchNorm2d(512)
        self.act_9 = Act_Q(bitA)
        
        self.conv_10 = Conv2d_CIM_SRAM(512, 512, kernel_size=3, padding=1, bias=False, bitW=bitW, bitO=bitO, adc_list=adc_list)
        self.bn_10 = nn.BatchNorm2d(512)
        self.act_10 = Act_Q(bitA)
        
        self.conv_11 = Conv2d_CIM_SRAM(512, 512, kernel_size=3, padding=1, bias=False, bitW=bitW, bitO=bitO, adc_list=adc_list)
        self.bn_11 = nn.BatchNorm2d(512)
        self.act_11 = Act_Q(bitA)
        
        self.conv_12 = Conv2d_CIM_SRAM(512, 512, kernel_size=3, padding=1, bias=False, bitW=bitW, bitO=bitO, adc_list=adc_list)
        self.bn_12 = nn.BatchNorm2d(512)
        self.act_12 = Act_Q(bitA)
        
        self.conv_13 = Conv2d_CIM_SRAM(512, 512, kernel_size=3, padding=1, bias=False, bitW=bitW, bitO=bitO, adc_list=adc_list)
        self.bn_13 = nn.BatchNorm2d(512)
        self.act_13 = Act_Q(bitA)
        
        self.linear_14 = Linear_Q(in_features=7*7*512, out_features=4096, bias=False, bitW=bitW) 
        self.act_14 = Act_Q(bitA)
        
        self.linear_15 = Linear_Q(in_features=4096, out_features=4096, bias=False, bitW=bitW)
        self.act_15 = Act_Q(bitA)
        
        self.linear_16 = Linear_Q(in_features=4096, out_features=num_classes, bias=False, bitW=bitW) 
            
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.dropout = nn.Dropout()
        
        if init_weights:
            self._initialize_weights()

    def forward(self, x_dict):
        x = self.conv_1(x_dict)
        x = self.bn_1(x)
        x = self.act_1(x)
        
        x = self.conv_2(x)
        x = self.bn_2(x)
        x = self.act_2(x)

        x = self.max_pool(x)
        
        x = self.conv_3(x)
        x = self.bn_3(x)
        x = self.act_3(x)

        x = self.conv_4(x)
        x = self.bn_4(x)
        x = self.act_4(x)

        x = self.max_pool(x)

        x = self.conv_5(x)
        x = self.bn_5(x)
        x = self.act_5(x)

        x = self.conv_6(x)
        x = self.bn_6(x)
        x = self.act_6(x)

        x = self.conv_7(x)
        x = self.bn_7(x)
        x = self.act_7(x)

        x = self.max_pool(x)

        x = self.conv_8(x)
        x = self.bn_8(x)
        x = self.act_8(x)

        x = self.conv_9(x)
        x = self.bn_9(x)
        x = self.act_9(x)

        x = self.conv_10(x)
        x = self.bn_10(x)
        x = self.act_10(x)

        x = self.max_pool(x)

        x = self.conv_11(x)
        x = self.bn_11(x)
        x = self.act_11(x)

        x = self.conv_12(x)
        x = self.bn_12(x)
        x = self.act_12(x)

        x = self.conv_13(x)
        x = self.bn_13(x)
        x = self.act_13(x)

        x = self.max_pool(x)['act']

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        x = self.linear_14(x)
        x = self.act_14(x)['act']
        x = self.dropout(x)
        
        x = self.linear_15(x)
        x = self.act_15(x)['act']
        x = self.dropout(x)
        
        x = self.linear_16(x)

        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def max_pool(self, inp_dict):
        maxpool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        oup = maxpool2d(inp_dict['act'])
        return {'act': oup, 'alpha': inp_dict['alpha']}

def _vgg(arch, cfg, batch_norm, bitW=32, bitA=32, bitO=32, sub_channel='v', adc_list=None, pretrained=False, **kwargs):
    model = VGG(bitW=bitW, bitA=bitA, bitO=bitO, adc_list=adc_list, **kwargs)
    if pretrained:
        kwargs['init_weights'] = False
        print(arch)

        # choose which state_dict to use
        #state_dict = load_state_dict_from_url(model_urls[arch],
        #                                     progress=True)
        model_best = torch.load('/home/u4416566/R2Q/01_train_from_scratch/logs/vgg16_sram_w8a8_sa32_sub_cv/model_best.pth.tar')
        state_dict = model_best['state_dict']
        
        model_dict = model.state_dict()
        
        #state_dict = {k: v for k, v in state_dict.items() if
        #              (k in model_dict) and (model_dict[k].shape == state_dict[k].shape)}
        #model_dict.update(state_dict)
        
        # For ckpt of vgg_SRAM_alpha
        new_model_dict = {}
        for i in range(len(state_dict)):
            new_model_dict[list(model_dict.keys())[i]] = list(state_dict.values())[i]
        
        model.load_state_dict(new_model_dict)
    return model

def vgg16_bn(bitW=32, bitA=32, bitO=32, sub_channel='v', adc_list=None, **kwargs):
    r"""VGG 16-layer model (configuration "D") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg16_bn', 'D', True, bitW, bitA, bitO, sub_channel, adc_list, **kwargs)
