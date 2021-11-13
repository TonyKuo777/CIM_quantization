import torch
import torch.nn as nn
#from .utils import load_state_dict_from_url
from torch.hub import load_state_dict_from_url
#------------------------------------------------------
#from utils.QLayer import *
#from utils.QLayer_PACT_SRAM import *
#from utils.QLayer_PACT_SRAM_test import *
#from utils.QLayer_PACT_SRAM_tanh_trainablealpha import *
#from utils.QLayer_PACT_SRAM_tanh import *
from utils.QLayer_PACT_SRAM_ste import *
#------------------------------------------------------

__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19',
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

    def __init__(self, features, bitW=32, bitA=32, num_classes=1000, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            #nn.Linear(512 * 7 * 7, 4096),
            #nn.ReLU(True),
            Linear_Q(in_features=7*7*512, out_features=4096, bias=False, bitW=bitW), 
            Act_Q(bitA=bitA),
            nn.Dropout(),
            #nn.Linear(4096, 4096),
            #nn.ReLU(True),
            Linear_Q(in_features=4096, out_features=4096, bias=False, bitW=bitW), 
            Act_Q(bitA=bitA),
            nn.Dropout(),
            #nn.Linear(4096, num_classes),
            Linear_Q(in_features=4096, out_features=num_classes, bias=False, bitW=bitW),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
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


def make_layers(cfg, batch_norm=False, bitW=32, bitA=32, bitO=32, sub_channel='v'):
    layers = []
    in_channels = 3
    first_layer = 1
    #comb_layer = 1

    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            if first_layer == 1:
                #conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1, bias=False)
                conv2d = Conv2d_SRAM(in_channels, v, kernel_size=3, padding=1, bias=False, bitW=bitW, bitO=bitO, sub_channel=sub_channel)
                first_layer = 0
            #elif comb_layer == 13:
            #    conv2d = Conv2d_CIM_SRAM_comb(in_channels, v, kernel_size=3, padding=1, bias=False, bitW=bitW, bitO=bitO, sub_channel=sub_channel)
            else:
                #conv2d = Conv2d_SRAM(in_channels, v, kernel_size=3, padding=1, bias=False, bitW=bitW, bitO=bitO, sub_channel=sub_channel)
                conv2d = Conv2d_CIM_SRAM(in_channels, v, kernel_size=3, padding=1, bias=False, bitW=bitW, bitO=bitO, sub_channel=sub_channel)
            #comb_layer += 1

            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), Act_Q(bitA)]
            else:
                layers += [conv2d, Act_Q(bitA)]
            in_channels = v
    return nn.Sequential(*layers)


cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

def _vgg(arch, cfg, batch_norm, bitW=32, bitA=32, bitO=32, sub_channel='v', pretrained=False, **kwargs):
    model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm, bitW=bitW, bitA=bitA, bitO=bitO, sub_channel=sub_channel), bitW=bitW, bitA=bitA, **kwargs)
    if pretrained:
        kwargs['init_weights'] = False
        print(arch)
               
        # choose which state_dict to use
        #state_dict = load_state_dict_from_url(model_urls[arch],
        #                                      progress=True)
        
        model_best = torch.load('/home/u4416566/R2Q/01_train_from_scratch/logs/vgg16_sram_ste_BLcomb_012_w8a8_sa32_sub_cv/checkpoint.pth.tar')
        
        k_prime_arr = torch.zeros(len(model_best['state_dict']))
        v_prime_arr = torch.zeros(len(model_best['state_dict']))
        k_prime_list = list(k_prime_arr)
        v_prime_list = list(v_prime_arr)
        i = 0
        for k, v in model_best['state_dict'].items():             # k: ['epoch'] ['state_dict'] ['best_acc1'] ['optimizer']
            rm_k = k[7:]
            k_prime_list[i] = rm_k
            v_prime_list[i] = v
            i += 1
        model_best['state_dict'] = {k_prime:v_prime for k_prime, v_prime in zip(k_prime_list, v_prime_list)}
        #state_dict = {k: v for k, v in checkpoint['state_dict'].items() if model.state_dict()[k].numel() == v.numel()}state_dict = model_best['state_dict']
        
        #model.load_state_dict(model_best['state_dict'])
        state_dict = model_best['state_dict']
        model_dict = model.state_dict()
        state_dict = {k: v for k, v in state_dict.items() if
                      (k in model_dict) and (model_dict[k].shape == state_dict[k].shape)}
        model_dict.update(state_dict)
        model.load_state_dict(model_dict)
    return model


def vgg11(bitW=32, bitA=32, bitO=32, sub_channel='v', **kwargs):
    r"""VGG 11-layer model (configuration "A") from
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg11', 'A', False, bitW, bitA, bitO, sub_channel, **kwargs)


def vgg11_bn(bitW=32, bitA=32, bitO=32, sub_channel='v', **kwargs):
    r"""VGG 11-layer model (configuration "A") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg11_bn', 'A', True, bitW, bitA, bitO, sub_channel, **kwargs)


def vgg13(bitW=32, bitA=32, bitO=32, sub_channel='v', **kwargs):
    r"""VGG 13-layer model (configuration "B")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg13', 'B', False, bitW, bitA, bitO, sub_channel, **kwargs)


def vgg13_bn(bitW=32, bitA=32, bitO=32, sub_channel='v', **kwargs):
    r"""VGG 13-layer model (configuration "B") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg13_bn', 'B', True, bitW, bitA, bitO, sub_channel, **kwargs)

def vgg16(bitW=32, bitA=32, bitO=32, sub_channel='v', **kwargs):
    r"""VGG 16-layer model (configuration "D")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg16', 'D', False, bitW, bitA, bitO, sub_channel, **kwargs)


def vgg16_bn(bitW=32, bitA=32, bitO=32, sub_channel='v', **kwargs):
    r"""VGG 16-layer model (configuration "D") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg16_bn', 'D', True, bitW, bitA, bitO, sub_channel, **kwargs)


def vgg19(bitW=32, bitA=32, bitO=32, sub_channel='v', **kwargs):
    r"""VGG 19-layer model (configuration "E")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg19', 'E', False, bitW, bitA, bitO, sub_channel, **kwargs)


def vgg19_bn(bitW=32, bitA=32, bitO=32, sub_channel='v', **kwargs):
    r"""VGG 19-layer model (configuration 'E') with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg19_bn', 'E', True, bitW, bitA, bitO, sub_channel, **kwargs)
