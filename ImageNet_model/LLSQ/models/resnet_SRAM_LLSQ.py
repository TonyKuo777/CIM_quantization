import math
import json
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict

from LLSQ_Modules.Conv2d_quan_ste_gpu import RoundFn_LLSQ, RoundFn_Bias
from LLSQ_Modules.Conv2d_quan_ste_gpu import QuantConv2d as Conv2dQ
from LLSQ_Modules.Conv2d_quan_ste_gpu import Conv2d_CIM_SRAM as Conv2dQ_cim
#from LLSQ_Modules.Conv2d_quan_ste_gen import RoundFn_LLSQ, RoundFn_Bias
#from LLSQ_Modules.Conv2d_quan_ste_gen import QuantConv2d as Conv2dQ
#from LLSQ_Modules.Conv2d_quan_ste_gen import Conv2d_CIM_SRAM as Conv2dQ_cim
from LLSQ_Modules.Quan_Act import RoundFn_act
from LLSQ_Modules.Quan_Act import ACT_Q as ActQ
from LLSQ_Modules.Linear_Q import Linear_Q

__all__ = ['ResNetQ',
           'resnet18_q', 'resnet18_qfn', 'resnet18_qfi',
           'resnet34_q',
           'resnet50_q', 'resnet50_qfn',
           'resnet18_qv2', 'resnet18_qfnv2',
           'resnet34_qv2', 'resnet34_qfnv2']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

def load_fake_quantized_state_dict(model, original_state_dict, key_map=None):
    if not isinstance(key_map, OrderedDict):
        with open('/home/u4416566/R2Q/01_train_from_scratch/LLSQ/models/{}'.format(key_map)) as rf:
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

def convq3x3(in_planes, out_planes, stride=1, nbits_w=4):
    """3x3 convolution with padding"""
    return Conv2dQ_cim(in_planes, out_planes, kernel_size=3, stride=stride,
                   padding=1, bias=False, bit=nbits_w)
    #return Conv2dQ(in_planes, out_planes, kernel_size=3, stride=stride,
    #               padding=1, bias=False, bit=nbits_w)

def convqv2_3x3(in_planes, out_planes, stride=1, nbits_w=4):
    """3x3 convolution with padding"""
    return Conv2dQv2(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False, bit=nbits_w)

class BasicBlockQ(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, nbits_w=4, nbits_a=4):
        super(BasicBlockQ, self).__init__()
        self.conv1 = nn.Sequential(convq3x3(inplanes, planes, stride, nbits_w=nbits_w),
                                   nn.BatchNorm2d(planes),
                                   nn.ReLU(inplace=True),
                                   ActQ(bit=nbits_a))
        self.conv2 = nn.Sequential(convq3x3(planes, planes, nbits_w=nbits_w),
                                   nn.BatchNorm2d(planes),
                                   ActQ(bit=nbits_a))
        self.downsample = downsample
        self.stride = stride
        self.relu = nn.ReLU(inplace=True)
        self.out_actq = ActQ(bit=nbits_a)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        out = self.out_actq(out)
        return out

class BasicBlockQv2(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, nbits_w=4, nbits_a=4):
        super(BasicBlockQv2, self).__init__()
        self.conv1 = nn.Sequential(convqv2_3x3(inplanes, planes, stride, nbits_w=nbits_w),
                                   nn.BatchNorm2d(planes),
                                   nn.ReLU(inplace=True),
                                   ActQv2(bit=nbits_a))
        self.conv2 = nn.Sequential(convqv2_3x3(planes, planes, nbits_w=nbits_w),
                                   nn.BatchNorm2d(planes),
                                   ActQv2(bit=nbits_a))
        self.downsample = downsample
        self.stride = stride
        self.relu = nn.ReLU(inplace=True)
        self.out_actq = ActQv2(bit=nbits_a)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        out = self.out_actq(out)
        return out

class BottleneckQ(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, nbits_w=4, nbits_a=4, alpha_bit=32, adc_list=None):
        super(BottleneckQ, self).__init__()
        self.conv1 = nn.Sequential(
            Conv2dQ_cim(inplanes, planes, kernel_size=1, bias=False, bit=nbits_w),
            #Conv2dQ(inplanes, planes, kernel_size=1, bias=False, bit=nbits_w),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
            ActQ(bit=nbits_a, alpha_bit=alpha_bit))
        self.conv2 = nn.Sequential(
            Conv2dQ_cim(planes, planes, kernel_size=3, stride=stride,
                    padding=1, bias=False, bit=nbits_w),
            #Conv2dQ(planes, planes, kernel_size=3, stride=stride,
            #        padding=1, bias=False, bit=nbits_w),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
            ActQ(bit=nbits_a, alpha_bit=alpha_bit))
        self.conv3 = nn.Sequential(
            Conv2dQ_cim(planes, planes * 4, kernel_size=1, bias=False, bit=nbits_w),
            #Conv2dQ(planes, planes * 4, kernel_size=1, bias=False, bit=nbits_w),
            nn.BatchNorm2d(planes * 4),
            ActQ(bit=nbits_a - 1, signed=True, alpha_bit=alpha_bit))
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.out_actq = ActQ(bit=nbits_a, alpha_bit=alpha_bit)
        self.stride = stride

    def forward(self, x_dict):
        residual = x_dict['act']

        out = self.conv1(x_dict)

        out = self.conv2(out)

        out = self.conv3(out)['act']

        if self.downsample is not None:
            residual = self.downsample(x_dict)['act']

        out += residual
        out = self.relu(out)
        out = self.out_actq(out)
        return out


class ResNetQ(nn.Module):
    def __init__(self, block, layers, num_classes=1000, nbits_w=4, nbits_a=4, alpha_bit=32, adc_list=None):
        self.inplanes = 64
        super(ResNetQ, self).__init__()
        # We don't quantize first layer
        self.nbits_w = nbits_w
        self.nbits_a = nbits_a
        self.alpha_bit = alpha_bit
        self.adc_list = adc_list
        self.conv1 = nn.Sequential(
            #nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            Conv2dQ(3, 64, kernel_size=7, stride=2, padding=3, bias=False, bit=nbits_w),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            ActQ(bit=self.nbits_a, signed=False, alpha_bit=alpha_bit),)
        #self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.Sequential(nn.AvgPool2d(7, stride=1), )
        self.fc = Linear_Q(512 * block.expansion, num_classes, bias=True, bit=nbits_w)
        #self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                Conv2dQ_cim(self.inplanes, planes * block.expansion,
                        kernel_size=1, stride=stride, bias=False,
                        bit=self.nbits_w),
                #Conv2dQ(self.inplanes, planes * block.expansion,
                #        kernel_size=1, stride=stride, bias=False,
                #        bit=self.nbits_w),
                nn.BatchNorm2d(planes * block.expansion),
                ActQ(bit=self.nbits_a - 1, signed=True, alpha_bit=self.alpha_bit),  # different with pre-trained model
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample,
                            nbits_w=self.nbits_w, nbits_a=self.nbits_a, alpha_bit=self.alpha_bit, adc_list=self.adc_list))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, nbits_w=self.nbits_w,
                                nbits_a=self.nbits_a, alpha_bit=self.alpha_bit, adc_list=self.adc_list))

        return nn.Sequential(*layers)
    
    def max_pool(self, inp_dict):
        maxpool2d = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        oup = maxpool2d(inp_dict['act'])
        return {'act': oup, 'alpha': inp_dict['alpha']}
    
    def forward(self, x):
        x = self.conv1(x)
        #x = self.maxpool(x)
        x = self.max_pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)['act']

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class ResNetQv2(nn.Module):

    def __init__(self, block, layers, num_classes=1000, nbits_w=4, nbits_a=4):
        self.inplanes = 64
        super(ResNetQv2, self).__init__()
        # We don't quantize first layer
        self.nbits_w = nbits_w
        self.nbits_a = nbits_a
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))  # del ActQ as LQ-Net
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.Sequential(nn.AvgPool2d(7, stride=1), )  # del ActQ as LQ-Net
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                Conv2dQv2(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False,
                          nbits=self.nbits_w),
                nn.BatchNorm2d(planes * block.expansion),
                ActQv2(bit=self.nbits_a, signed=True),  # different with pre-trained model
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample,
                            nbits_w=self.nbits_w, nbits_a=self.nbits_a))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, nbits_w=self.nbits_w,
                                nbits_a=self.nbits_a))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class ResNetQFNv2(nn.Module):

    def __init__(self, block, layers, num_classes=1000, nbits_w=4, nbits_a=4):
        self.inplanes = 64
        super(ResNetQFNv2, self).__init__()
        # We don't quantize first layer
        self.nbits_w = nbits_w
        self.nbits_a = nbits_a
        self.conv1 = nn.Sequential(
            ActQv2(bit=-1 if max(nbits_a, nbits_w) <= 0 else 8, signed=True),
            Conv2dQv2(3, 64, kernel_size=7, stride=2, padding=3, bias=False, bit=nbits_w),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))  # del ActQ as LQ-Net
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        # self.avgpool = nn.Sequential(nn.AvgPool2d(7, stride=1), )  # del ActQ as LQ-Net
        # self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.avgpool = nn.Sequential(nn.AvgPool2d(7, stride=1),
                                     ActQv2(bit=nbits_a))  # del ActQ as LQ-Net
        self.fc = LinearQv2(512 * block.expansion, num_classes, bit=nbits_w)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                Conv2dQv2(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False,
                          bit=self.nbits_w),
                nn.BatchNorm2d(planes * block.expansion),
                ActQv2(bit=self.nbits_a, signed=True),  # different with pre-trained model
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample,
                            nbits_w=self.nbits_w, nbits_a=self.nbits_a))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, nbits_w=self.nbits_w,
                                nbits_a=self.nbits_a))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class ResNetQFI(nn.Module):

    def __init__(self, block, layers, num_classes=1000, nbits_w=4, nbits_a=4):
        self.inplanes = 64
        super(ResNetQFI, self).__init__()
        self.nbits_w = nbits_w
        self.nbits_a = nbits_a
        self.conv1 = nn.Sequential(
            ActQ(bit=-1 if max(nbits_a, nbits_w) <= 0 else 8, signed=True),
            Conv2dQ_cim(3, 64, kernel_size=7, stride=2, padding=3, bias=False, bit=nbits_w),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))  # del ActQ as LQ-Net
        self.maxpool = nn.Sequential(nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                                     ActQ(bit=nbits_a))
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)

        self.expand_actq = ActQ(bit=-1 if max(nbits_a, nbits_w) <= 0 else 8,
                                expand=True)
        self.expand_fc = LinearQ(512 * block.expansion * 2, num_classes, bit=nbits_w)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                Conv2dQ_cim(self.inplanes, planes * block.expansion,
                        kernel_size=1, stride=stride, bias=False,
                        bit=self.nbits_w),
                nn.BatchNorm2d(planes * block.expansion),
                ActQ(bit=self.nbits_a, signed=True),  # different with pre-trained model
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample,
                            nbits_w=self.nbits_w, nbits_a=self.nbits_a))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, nbits_w=self.nbits_w,
                                nbits_a=self.nbits_a))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.expand_actq(x)
        x = self.expand_fc(x)
        return x


class ResNetQFN(nn.Module):

    def __init__(self, block, layers, num_classes=1000, nbits_w=4, nbits_a=4):
        self.inplanes = 64
        super(ResNetQFN, self).__init__()
        self.nbits_w = nbits_w
        self.nbits_a = nbits_a
        self.conv1 = nn.Sequential(
            ActQ(bit=-1 if max(nbits_a, nbits_w) <= 0 else 8, signed=False),
            Conv2dQ_cim(3, 64, kernel_size=7, stride=2, padding=3, bias=False, bit=nbits_w),
            #nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))  # del ActQ as LQ-Net
        self.maxpool = nn.Sequential(nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                                     ActQ(bit=nbits_a))
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.Sequential(nn.AvgPool2d(7, stride=1),
                                     ActQ(bit=nbits_a))  # del ActQ as LQ-Net
        self.fc = Linear_Q(512 * block.expansion, num_classes, bit=nbits_w)
        #self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                Conv2dQ_cim(self.inplanes, planes * block.expansion,
                        kernel_size=1, stride=stride, bias=False,
                        bit=self.nbits_w),
                nn.BatchNorm2d(planes * block.expansion),
                ActQ(bit=self.nbits_a, signed=False),  # different with pre-trained model
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample,
                            nbits_w=self.nbits_w, nbits_a=self.nbits_a))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, nbits_w=self.nbits_w,
                                nbits_a=self.nbits_a))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet18_qv2(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNetQv2(BasicBlockQv2, [2, 2, 2, 2], **kwargs)
    if pretrained:
        load_fake_quantized_state_dict(model, model_zoo.load_url(model_urls['resnet18']), 'resnet18_qv2_map.json')
    return model


def resnet18_qfnv2(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNetQFNv2(BasicBlockQv2, [2, 2, 2, 2], **kwargs)
    if pretrained:
        load_fake_quantized_state_dict(model, model_zoo.load_url(model_urls['resnet18']), 'resnet18_qfnv2_map.json')
    return model


def resnet18_q(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNetQ(BasicBlockQ, [2, 2, 2, 2], **kwargs)
    if pretrained:
        load_fake_quantized_state_dict(model, model_zoo.load_url(model_urls['resnet18']), 'resnet18_q_map.json')
    return model


def resnet18_qfn(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNetQFN(BasicBlockQ, [2, 2, 2, 2], **kwargs)
    if pretrained:
        load_fake_quantized_state_dict(model, model_zoo.load_url(model_urls['resnet18']), 'resnet18_qfn_map.json')
    return model


def resnet18_qfi(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNetQFI(BasicBlockQ, [2, 2, 2, 2], **kwargs)
    if pretrained:
        load_fake_quantized_state_dict(model, model_zoo.load_url(model_urls['resnet18']), 'resnet18_qfi_map.json')
    return model


def resnet34_q(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNetQ(BasicBlockQ, [3, 4, 6, 3], **kwargs)
    if pretrained:
        load_fake_quantized_state_dict(model, model_zoo.load_url(model_urls['resnet34']), 'resnet34_q_map.json')
    return model


def resnet34_qv2(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNetQv2(BasicBlockQv2, [3, 4, 6, 3], **kwargs)
    if pretrained:
        load_fake_quantized_state_dict(model, model_zoo.load_url(model_urls['resnet34']), 'resnet34_qv2_map.json')
    return model


def resnet34_qfnv2(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNetQFNv2(BasicBlockQv2, [3, 4, 6, 3], **kwargs)
    if pretrained:
        load_fake_quantized_state_dict(model, model_zoo.load_url(model_urls['resnet34']), 'resnet34_qfnv2_map.json')
    return model


def resnet34_qfi(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNetQFI(BasicBlockQ, [3, 4, 6, 3], **kwargs)
    if pretrained:
        load_fake_quantized_state_dict(model, model_zoo.load_url(model_urls['resnet34']), 'resnet34_qfi_map.json')
    return model


def resnet50_q(nbits_w=32, nbits_a=32, alpha_bit=32, pretrained=False, adc_list=None, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNetQ(BottleneckQ, [3, 4, 6, 3], num_classes=1000, nbits_w=nbits_w, nbits_a=nbits_a, alpha_bit=alpha_bit, adc_list=adc_list, **kwargs)
    if pretrained:
        load_fake_quantized_state_dict(model, model_zoo.load_url(model_urls['resnet50']), 'resnet50_q_map.json')
    return model


def resnet50_qfn(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNetQFN(BottleneckQ, [3, 4, 6, 3], **kwargs)
    if pretrained:
        load_fake_quantized_state_dict(model, model_zoo.load_url(model_urls['resnet50']), 'resnet50_qfn_map.json')
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNetQ(BottleneckQ, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNetQ(BottleneckQ, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model
