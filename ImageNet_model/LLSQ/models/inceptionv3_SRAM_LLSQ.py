import json
from collections import namedtuple, OrderedDict
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.model_zoo import load_url as load_state_dict_from_url
import torch.utils.model_zoo as model_zoo
from torchvision import models
from typing import Callable, Any, Optional, Tuple, List
#---------------------------------------
from LLSQ_Modules.Conv2d_quan_ste_gpu import RoundFn_LLSQ, RoundFn_Bias
from LLSQ_Modules.Conv2d_quan_ste_gpu import QuantConv2d as Conv2d_quan
from LLSQ_Modules.Conv2d_quan_ste_gpu import Conv2d_CIM_SRAM as Conv2d_quan_cim
from LLSQ_Modules.Quan_Act import RoundFn_act, ACT_Q
from LLSQ_Modules.Linear_Q import Linear_Q
#---------------------------------------

__all__ = ['Inception3Q', 'inceptionv3_q', 'InceptionOutputs', '_InceptionOutputs']


model_urls = {
    # Inception v3 ported from TensorFlow
    'inception_v3_google': 'https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth',
}

InceptionOutputs = namedtuple('InceptionOutputs', ['logits', 'aux_logits'])
InceptionOutputs.__annotations__ = {'logits': torch.Tensor, 'aux_logits': Optional[torch.Tensor]}

# Script annotations failed with _GoogleNetOutputs = namedtuple ...
# _InceptionOutputs set here for backwards compat
_InceptionOutputs = InceptionOutputs

def load_fake_quantized_state_dict(model, checkpoint, key_map=None):
    
    k_prime_arr = torch.zeros(len(checkpoint['state_dict']))
    v_prime_arr = torch.zeros(len(checkpoint['state_dict']))
    k_prime_list = list(k_prime_arr)
    v_prime_list = list(v_prime_arr)
    i = 0
    for k, v in checkpoint['state_dict'].items():             # k: ['epoch'] ['state_dict'] ['best_acc1'] ['optimizer']
        rm_k = k[7:]
        k_prime_list[i] = rm_k
        v_prime_list[i] = v
        i += 1
    checkpoint['state_dict'] = {k_prime:v_prime for k_prime, v_prime in zip(k_prime_list, v_prime_list)}
    pretrained_state_dict = checkpoint['state_dict']
    
    if not isinstance(key_map, OrderedDict):
        with open('/home/u4416566/R2Q/01_train_from_scratch/LLSQ/models/{}'.format(key_map)) as rf:
            key_map = json.load(rf)
    for k, v in key_map.items():
        if 'num_batches_tracked' in k:
            continue
        if 'expand_' in k and model.state_dict()[k].shape != pretrained_state_dict[v].shape:
            ori_weight = pretrained_state_dict[v]
            new_weight = torch.cat((ori_weight, ori_weight * 2 ** 4), dim=1)
            model.state_dict()[k].copy_(new_weight)
        else:
            model.state_dict()[k].copy_(pretrained_state_dict[v])

def inceptionv3_q(nbits_w=32, nbits_a=32, pretrained: bool = False, progress: bool = True, adc_list=None, **kwargs: Any) -> "Inception3":
    r"""Inception v3 model architecture from
    `"Rethinking the Inception Architecture for Computer Vision" <http://arxiv.org/abs/1512.00567>`_.
    .. note::
        **Important**: In contrast to the other models the inception_v3 expects tensors with a size of
        N x 3 x 299 x 299, so ensure your images are sized accordingly.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        aux_logits (bool): If True, add an auxiliary branch that can improve training.
            Default: *True*
        transform_input (bool): If True, preprocesses the input according to the method with which it
            was trained on ImageNet. Default: *False*
    """
    model_name = 'inception_v3'
    if 'transform_input' not in kwargs:
            kwargs['transform_input'] = True
    if pretrained:
        if 'aux_logits' in kwargs:
            original_aux_logits = kwargs['aux_logits']
            kwargs['aux_logits'] = True
        else:
            original_aux_logits = True
        kwargs['init_weights'] = False  # we are loading weights from a pretrained model
        model = Inception3Q(nbits_w=nbits_w, nbits_a=nbits_a, **kwargs)
        
        #load_fake_quantized_state_dict(model, model_zoo.load_url(model_urls['inception_v3_google'], map_location='cpu'),
        #                               'inception_v3_map.json')
        checkpoint = torch.load('/home/u4416566/R2Q/01_train_from_scratch/LLSQ/logs/inceptionv3_sram_llsq_TWCC_w8a9_sa32_sub_cv/model_best.pth.tar')
        load_fake_quantized_state_dict(model, checkpoint,
                                       'inception_v3_map.json')
        if not original_aux_logits:
            model.aux_logits = False
            model.AuxLogits = None
        return model, model_name

    return Inception3Q(nbits_w=nbits_w, nbits_a=nbits_a, adc_list=adc_list, **kwargs), model_name


class Inception3Q(nn.Module):

    def __init__(
        self,
        nbits_w=32, nbits_a=32,
        adc_list=None,
        num_classes: int = 1000,
        aux_logits: bool = True,
        transform_input: bool = False,
        inception_blocks: Optional[List[Callable[..., nn.Module]]] = None,
        init_weights: Optional[bool] = None
    ) -> None:
        super(Inception3Q, self).__init__()
        if inception_blocks is None:
            #inception_blocks = [
            #    BasicConv2dQ, InceptionAQ, InceptionBQ, InceptionCQ,
            #    InceptionDQ, InceptionEQ, InceptionAuxQ
            #]
            inception_blocks = [
                BasicConv2dQ_first_layer, BasicConv2dQ, InceptionAQ, InceptionBQ, InceptionCQ,
                InceptionDQ, InceptionEQ, InceptionAuxQ
            ]
        if init_weights is None:
            warnings.warn('The default weight initialization of inception_v3 will be changed in future releases of '
                          'torchvision. If you wish to keep the old behavior (which leads to long initialization times'
                          ' due to scipy/scipy#11299), please set init_weights=True.', FutureWarning)
            init_weights = True
        #assert len(inception_blocks) == 7
        #conv_block = inception_blocks[0]
        #inception_a = inception_blocks[1]
        #inception_b = inception_blocks[2]
        #inception_c = inception_blocks[3]
        #inception_d = inception_blocks[4]
        #inception_e = inception_blocks[5]
        #inception_aux = inception_blocks[6]
        assert len(inception_blocks) == 8
        conv_block_first_layer = inception_blocks[0]
        conv_block = inception_blocks[1]
        inception_a = inception_blocks[2]
        inception_b = inception_blocks[3]
        inception_c = inception_blocks[4]
        inception_d = inception_blocks[5]
        inception_e = inception_blocks[6]
        inception_aux = inception_blocks[7]


        self.aux_logits = aux_logits
        self.transform_input = transform_input
        self.Conv2d_1a_3x3 = conv_block_first_layer(3, 32, kernel_size=3, stride=2, nbits_w=nbits_w, nbits_a=nbits_a)
        self.Conv2d_2a_3x3 = conv_block(32, 32, kernel_size=3, nbits_w=nbits_w, nbits_a=nbits_a, adc_list=adc_list)
        self.Conv2d_2b_3x3 = conv_block(32, 64, kernel_size=3, padding=1, nbits_w=nbits_w, nbits_a=nbits_a, adc_list=adc_list)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.Conv2d_3b_1x1 = conv_block(64, 80, kernel_size=1, nbits_w=nbits_w, nbits_a=nbits_a, adc_list=adc_list)
        self.Conv2d_4a_3x3 = conv_block(80, 192, kernel_size=3, nbits_w=nbits_w, nbits_a=nbits_a, adc_list=adc_list)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.Mixed_5b = inception_a(192, pool_features=32, nbits_w=nbits_w, nbits_a=nbits_a, adc_list=adc_list)
        self.Mixed_5c = inception_a(256, pool_features=64, nbits_w=nbits_w, nbits_a=nbits_a, adc_list=adc_list)
        self.Mixed_5d = inception_a(288, pool_features=64, nbits_w=nbits_w, nbits_a=nbits_a, adc_list=adc_list)
        self.Mixed_6a = inception_b(288, nbits_w=nbits_w, nbits_a=nbits_a, adc_list=adc_list)
        self.Mixed_6b = inception_c(768, channels_7x7=128, nbits_w=nbits_w, nbits_a=nbits_a, adc_list=adc_list)
        self.Mixed_6c = inception_c(768, channels_7x7=160, nbits_w=nbits_w, nbits_a=nbits_a, adc_list=adc_list)
        self.Mixed_6d = inception_c(768, channels_7x7=160, nbits_w=nbits_w, nbits_a=nbits_a, adc_list=adc_list)
        self.Mixed_6e = inception_c(768, channels_7x7=192, nbits_w=nbits_w, nbits_a=nbits_a, adc_list=adc_list)
        self.AuxLogits: Optional[nn.Module] = None
        if aux_logits:
            self.AuxLogits = inception_aux(768, num_classes, nbits_w=nbits_w, nbits_a=nbits_a, adc_list=adc_list)
        self.Mixed_7a = inception_d(768, nbits_w=nbits_w, nbits_a=nbits_a, adc_list=adc_list)
        self.Mixed_7b = inception_e(1280, nbits_w=nbits_w, nbits_a=nbits_a, adc_list=adc_list)
        self.Mixed_7c = inception_e(2048, nbits_w=nbits_w, nbits_a=nbits_a, adc_list=adc_list)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout()
        #self.fc = nn.Linear(2048, num_classes)
        self.fc = Linear_Q(2048, num_classes, bias=True, bit=nbits_w)
        if init_weights:
            for m in self.modules():
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                    import scipy.stats as stats
                    stddev = m.stddev if hasattr(m, 'stddev') else 0.1
                    X = stats.truncnorm(-2, 2, scale=stddev)
                    values = torch.as_tensor(X.rvs(m.weight.numel()), dtype=m.weight.dtype)
                    values = values.view(m.weight.size())
                    with torch.no_grad():
                        m.weight.copy_(values)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def _transform_input(self, x: Tensor) -> Tensor:
        if self.transform_input:
            x_ch0 = torch.unsqueeze(x[:, 0], 1) * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x_ch1 = torch.unsqueeze(x[:, 1], 1) * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x_ch2 = torch.unsqueeze(x[:, 2], 1) * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
            x = torch.cat((x_ch0, x_ch1, x_ch2), 1)
            return x

    def _forward(self, x: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
        # N x 3 x 299 x 299
        x = self.Conv2d_1a_3x3(x)
        # N x 32 x 149 x 149
        x = self.Conv2d_2a_3x3(x)
        # N x 32 x 147 x 147
        x = self.Conv2d_2b_3x3(x)
        # N x 64 x 147 x 147
        #x = self.maxpool1(x)
        x = self.max_pool(x)
        # N x 64 x 73 x 73
        x = self.Conv2d_3b_1x1(x)
        # N x 80 x 73 x 73
        x = self.Conv2d_4a_3x3(x)
        # N x 192 x 71 x 71
        #x = self.maxpool2(x)
        x = self.max_pool(x)
        # N x 192 x 35 x 35
        x = self.Mixed_5b(x)
        # N x 256 x 35 x 35
        x = self.Mixed_5c(x)
        # N x 288 x 35 x 35
        x = self.Mixed_5d(x)
        # N x 288 x 35 x 35
        x = self.Mixed_6a(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6b(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6c(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6d(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6e(x)
        # N x 768 x 17 x 17
        aux = torch.jit.annotate(Optional[Tensor], None)
        if self.AuxLogits and self.training:
            aux = self.AuxLogits(x)                       # class <'torch.Tensor'>
        # N x 768 x 17 x 17
        x = self.Mixed_7a(x)
        # N x 1280 x 8 x 8
        x = self.Mixed_7b(x)
        # N x 2048 x 8 x 8
        x = self.Mixed_7c(x)['act']
        # N x 2048 x 8 x 8
        # Adaptive average pooling
        x = self.avgpool(x)
        # N x 2048 x 1 x 1
        x = self.dropout(x)
        # N x 2048 x 1 x 1
        x = torch.flatten(x, 1)
        # N x 2048
        x = self.fc(x)
        # N x 1000 (num_classes)
        #if self.AuxLogits and self.training:
        #    return x, aux
        return x, aux

    @torch.jit.unused
    def eager_outputs(self, x: Tensor, aux: Optional[Tensor]) -> InceptionOutputs:
        if self.training and self.aux_logits:
            return x, aux
            #return InceptionOutputs(x, aux)
        else:
            return x  # type: ignore[return-value]

    def max_pool(self, inp_dict):
        maxpool2d = nn.MaxPool2d(kernel_size=3, stride=2)
        oup = maxpool2d(inp_dict['act'])
        return {'act': oup, 'alpha': inp_dict['alpha']}
    
    def forward(self, x: Tensor) -> InceptionOutputs:
        x = self._transform_input(x)
        x, aux = self._forward(x)
        aux_defined = self.training and self.aux_logits
        if torch.jit.is_scripting():
            if not aux_defined:
                warnings.warn("Scripted Inception3 always returns Inception3 Tuple")
            return x, aux
            #return InceptionOutputs(x, aux)
        else:
            return self.eager_outputs(x, aux)


class InceptionAQ(nn.Module):

    def __init__(
        self,
        in_channels: int,
        pool_features: int,
        nbits_w=32, nbits_a=32,
        adc_list=None,
        conv_block: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(InceptionAQ, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2dQ
        self.branch1x1 = conv_block(in_channels, 64, kernel_size=1, nbits_w=nbits_w, nbits_a=nbits_a, adc_list=adc_list)

        self.branch5x5_1 = conv_block(in_channels, 48, kernel_size=1, nbits_w=nbits_w, nbits_a=nbits_a, adc_list=adc_list)
        self.branch5x5_2 = conv_block(48, 64, kernel_size=5, padding=2, nbits_w=nbits_w, nbits_a=nbits_a, adc_list=adc_list)

        self.branch3x3dbl_1 = conv_block(in_channels, 64, kernel_size=1, nbits_w=nbits_w, nbits_a=nbits_a, adc_list=adc_list)
        self.branch3x3dbl_2 = conv_block(64, 96, kernel_size=3, padding=1, nbits_w=nbits_w, nbits_a=nbits_a, adc_list=adc_list)
        self.branch3x3dbl_3 = conv_block(96, 96, kernel_size=3, padding=1, nbits_w=nbits_w, nbits_a=nbits_a, adc_list=adc_list)

        self.branch_pool = conv_block(in_channels, pool_features, kernel_size=1, nbits_w=nbits_w, nbits_a=nbits_a, adc_list=adc_list)

    def _forward(self, x_dict: dict) -> List[dict]:
        branch1x1 = self.branch1x1(x_dict)
        
        branch5x5 = self.branch5x5_1(x_dict)
        branch5x5 = self.branch5x5_2(branch5x5)
        
        branch3x3dbl = self.branch3x3dbl_1(x_dict)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)
        
        #branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.avg_pool(x_dict)
        branch_pool = self.branch_pool(branch_pool)
        
        outputs_act = [branch1x1['act'], branch5x5['act'], branch3x3dbl['act'], branch_pool['act']]
        outputs_alpha = torch.max(torch.tensor([branch1x1['alpha'].data, branch5x5['alpha'].data, branch3x3dbl['alpha'].data, branch_pool['alpha'].data]))
        return {'act': outputs_act, 'alpha': outputs_alpha}
    
    def avg_pool(self, inp_dict):
        oup = F.avg_pool2d(inp_dict['act'], kernel_size=3, stride=1, padding=1)
        return {'act': oup, 'alpha': inp_dict['alpha']}

    def forward(self, x_dict: dict) -> dict:
        outputs_dict = self._forward(x_dict)
        return {'act': torch.cat(outputs_dict['act'], 1), 'alpha': outputs_dict['alpha']}


class InceptionBQ(nn.Module):

    def __init__(
        self,
        in_channels: int,
        nbits_w=32, nbits_a=32,
        adc_list=None,
        conv_block: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(InceptionBQ, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2dQ
        self.branch3x3 = conv_block(in_channels, 384, kernel_size=3, stride=2, nbits_w=nbits_w, nbits_a=nbits_a, adc_list=adc_list)

        self.branch3x3dbl_1 = conv_block(in_channels, 64, kernel_size=1, nbits_w=nbits_w, nbits_a=nbits_a, adc_list=adc_list)
        self.branch3x3dbl_2 = conv_block(64, 96, kernel_size=3, padding=1, nbits_w=nbits_w, nbits_a=nbits_a, adc_list=adc_list)
        self.branch3x3dbl_3 = conv_block(96, 96, kernel_size=3, stride=2, nbits_w=nbits_w, nbits_a=nbits_a, adc_list=adc_list)

    def _forward(self, x_dict: dict) -> List[dict]:
        branch3x3 = self.branch3x3(x_dict)

        branch3x3dbl = self.branch3x3dbl_1(x_dict)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        #branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)
        branch_pool = self.max_pool(x_dict)

        outputs_act = [branch3x3['act'], branch3x3dbl['act'], branch_pool['act']]
        outputs_alpha = torch.max(torch.tensor([branch3x3['alpha'].data, branch3x3dbl['alpha'].data, branch_pool['alpha'].data]))
        return {'act': outputs_act, 'alpha': outputs_alpha}
    
    def max_pool(self, inp_dict):
        oup = F.max_pool2d(inp_dict['act'], kernel_size=3, stride=2)
        return {'act': oup, 'alpha': inp_dict['alpha']}

    def forward(self, x_dict: dict) -> dict:
        outputs_dict = self._forward(x_dict)
        return {'act': torch.cat(outputs_dict['act'], 1), 'alpha': outputs_dict['alpha']}


class InceptionCQ(nn.Module):

    def __init__(
        self,
        in_channels: int,
        channels_7x7: int,
        nbits_w=32, nbits_a=32,
        adc_list=None,
        conv_block: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(InceptionCQ, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2dQ
        self.branch1x1 = conv_block(in_channels, 192, kernel_size=1, nbits_w=nbits_w, nbits_a=nbits_a, adc_list=adc_list)

        c7 = channels_7x7
        self.branch7x7_1 = conv_block(in_channels, c7, kernel_size=1, nbits_w=nbits_w, nbits_a=nbits_a, adc_list=adc_list)
        self.branch7x7_2 = conv_block(c7, c7, kernel_size=(1, 7), padding=(0, 3), nbits_w=nbits_w, nbits_a=nbits_a, adc_list=adc_list)
        self.branch7x7_3 = conv_block(c7, 192, kernel_size=(7, 1), padding=(3, 0), nbits_w=nbits_w, nbits_a=nbits_a, adc_list=adc_list)

        self.branch7x7dbl_1 = conv_block(in_channels, c7, kernel_size=1, nbits_w=nbits_w, nbits_a=nbits_a, adc_list=adc_list)
        self.branch7x7dbl_2 = conv_block(c7, c7, kernel_size=(7, 1), padding=(3, 0), nbits_w=nbits_w, nbits_a=nbits_a, adc_list=adc_list)
        self.branch7x7dbl_3 = conv_block(c7, c7, kernel_size=(1, 7), padding=(0, 3), nbits_w=nbits_w, nbits_a=nbits_a, adc_list=adc_list)
        self.branch7x7dbl_4 = conv_block(c7, c7, kernel_size=(7, 1), padding=(3, 0), nbits_w=nbits_w, nbits_a=nbits_a, adc_list=adc_list)
        self.branch7x7dbl_5 = conv_block(c7, 192, kernel_size=(1, 7), padding=(0, 3), nbits_w=nbits_w, nbits_a=nbits_a, adc_list=adc_list)

        self.branch_pool = conv_block(in_channels, 192, kernel_size=1, nbits_w=nbits_w, nbits_a=nbits_a, adc_list=adc_list)

    def _forward(self, x_dict: dict) -> List[dict]:
        branch1x1 = self.branch1x1(x_dict)

        branch7x7 = self.branch7x7_1(x_dict)
        branch7x7 = self.branch7x7_2(branch7x7)
        branch7x7 = self.branch7x7_3(branch7x7)

        branch7x7dbl = self.branch7x7dbl_1(x_dict)
        branch7x7dbl = self.branch7x7dbl_2(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_3(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_4(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_5(branch7x7dbl)

        #branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.avg_pool(x_dict)
        branch_pool = self.branch_pool(branch_pool)

        outputs_act = [branch1x1['act'], branch7x7['act'], branch7x7dbl['act'], branch_pool['act']]
        outputs_alpha = torch.max(torch.tensor([branch1x1['alpha'].data, branch7x7['alpha'].data, branch7x7dbl['alpha'].data, branch_pool['alpha'].data]))
        return {'act': outputs_act, 'alpha': outputs_alpha}

    def avg_pool(self, inp_dict):
        oup = F.avg_pool2d(inp_dict['act'], kernel_size=3, stride=1, padding=1)
        return {'act': oup, 'alpha': inp_dict['alpha']}

    def forward(self, x_dict: dict) -> dict:
        outputs_dict = self._forward(x_dict)
        return {'act': torch.cat(outputs_dict['act'], 1), 'alpha': outputs_dict['alpha']}


class InceptionDQ(nn.Module):

    def __init__(
        self,
        in_channels: int,
        nbits_w=32, nbits_a=32,
        adc_list=None,
        conv_block: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(InceptionDQ, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2dQ
        self.branch3x3_1 = conv_block(in_channels, 192, kernel_size=1, nbits_w=nbits_w, nbits_a=nbits_a, adc_list=adc_list)
        self.branch3x3_2 = conv_block(192, 320, kernel_size=3, stride=2, nbits_w=nbits_w, nbits_a=nbits_a, adc_list=adc_list)

        self.branch7x7x3_1 = conv_block(in_channels, 192, kernel_size=1, nbits_w=nbits_w, nbits_a=nbits_a, adc_list=adc_list)
        self.branch7x7x3_2 = conv_block(192, 192, kernel_size=(1, 7), padding=(0, 3), nbits_w=nbits_w, nbits_a=nbits_a, adc_list=adc_list)
        self.branch7x7x3_3 = conv_block(192, 192, kernel_size=(7, 1), padding=(3, 0), nbits_w=nbits_w, nbits_a=nbits_a, adc_list=adc_list)
        self.branch7x7x3_4 = conv_block(192, 192, kernel_size=3, stride=2, nbits_w=nbits_w, nbits_a=nbits_a, adc_list=adc_list)

    def _forward(self, x_dict: dict) -> List[dict]:
        branch3x3 = self.branch3x3_1(x_dict)
        branch3x3 = self.branch3x3_2(branch3x3)

        branch7x7x3 = self.branch7x7x3_1(x_dict)
        branch7x7x3 = self.branch7x7x3_2(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_3(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_4(branch7x7x3)

        #branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)
        branch_pool = self.max_pool(x_dict)
        
        outputs_act = [branch3x3['act'], branch7x7x3['act'], branch_pool['act']]
        outputs_alpha = torch.max(torch.tensor([branch3x3['alpha'].data, branch7x7x3['alpha'].data, branch_pool['alpha'].data]))
        return {'act': outputs_act, 'alpha': outputs_alpha}

    def max_pool(self, inp_dict):
        oup = F.max_pool2d(inp_dict['act'], kernel_size=3, stride=2)
        return {'act': oup, 'alpha': inp_dict['alpha']}

    def forward(self, x_dict: dict) -> dict:
        outputs_dict = self._forward(x_dict)
        return {'act': torch.cat(outputs_dict['act'], 1), 'alpha': outputs_dict['alpha']}


class InceptionEQ(nn.Module):

    def __init__(
        self,
        in_channels: int,
        nbits_w=32, nbits_a=32,
        adc_list=None,
        conv_block: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(InceptionEQ, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2dQ
        self.branch1x1 = conv_block(in_channels, 320, kernel_size=1, nbits_w=nbits_w, nbits_a=nbits_a, adc_list=adc_list)

        self.branch3x3_1 = conv_block(in_channels, 384, kernel_size=1, nbits_w=nbits_w, nbits_a=nbits_a, adc_list=adc_list)
        self.branch3x3_2a = conv_block(384, 384, kernel_size=(1, 3), padding=(0, 1), nbits_w=nbits_w, nbits_a=nbits_a, adc_list=adc_list)
        self.branch3x3_2b = conv_block(384, 384, kernel_size=(3, 1), padding=(1, 0), nbits_w=nbits_w, nbits_a=nbits_a, adc_list=adc_list)

        self.branch3x3dbl_1 = conv_block(in_channels, 448, kernel_size=1, nbits_w=nbits_w, nbits_a=nbits_a, adc_list=adc_list)
        self.branch3x3dbl_2 = conv_block(448, 384, kernel_size=3, padding=1, nbits_w=nbits_w, nbits_a=nbits_a, adc_list=adc_list)
        self.branch3x3dbl_3a = conv_block(384, 384, kernel_size=(1, 3), padding=(0, 1), nbits_w=nbits_w, nbits_a=nbits_a, adc_list=adc_list)
        self.branch3x3dbl_3b = conv_block(384, 384, kernel_size=(3, 1), padding=(1, 0), nbits_w=nbits_w, nbits_a=nbits_a, adc_list=adc_list)

        self.branch_pool = conv_block(in_channels, 192, kernel_size=1, nbits_w=nbits_w, nbits_a=nbits_a, adc_list=adc_list)

    def _forward(self, x_dict: dict) -> List[dict]:
        branch1x1 = self.branch1x1(x_dict)

        branch3x3 = self.branch3x3_1(x_dict)
        branch3x3_act = [
            self.branch3x3_2a(branch3x3)['act'],
            self.branch3x3_2b(branch3x3)['act'],
        ]
        branch3x3_alpha = torch.max(torch.tensor([self.branch3x3_2a(branch3x3)['alpha'].data, self.branch3x3_2b(branch3x3)['alpha'].data]))
        branch3x3 = {'act': torch.cat(branch3x3_act, 1), 'alpha': branch3x3_alpha}

        branch3x3dbl = self.branch3x3dbl_1(x_dict)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl_act = [
            self.branch3x3dbl_3a(branch3x3dbl)['act'],
            self.branch3x3dbl_3b(branch3x3dbl)['act'],
        ]
        branch3x3dbl_alpha = torch.max(torch.tensor([self.branch3x3dbl_3a(branch3x3dbl)['alpha'].data, self.branch3x3dbl_3b(branch3x3dbl)['alpha'].data]))
        branch3x3dbl = {'act': torch.cat(branch3x3dbl_act, 1), 'alpha': branch3x3dbl_alpha}
        
        #branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.avg_pool(x_dict)
        branch_pool = self.branch_pool(branch_pool)

        outputs_act = [branch1x1['act'], branch3x3['act'], branch3x3dbl['act'], branch_pool['act']]
        outputs_alpha = torch.max(torch.tensor([branch1x1['alpha'].data, branch3x3['alpha'].data, branch3x3dbl['alpha'].data, branch_pool['alpha'].data]))
        return {'act': outputs_act, 'alpha': outputs_alpha}

    def avg_pool(self, inp_dict):
        oup = F.avg_pool2d(inp_dict['act'], kernel_size=3, stride=1, padding=1)
        return {'act': oup, 'alpha': inp_dict['alpha']}

    def forward(self, x_dict: dict) -> dict:
        outputs_dict = self._forward(x_dict)
        return {'act': torch.cat(outputs_dict['act'], 1), 'alpha': outputs_dict['alpha']}


class InceptionAuxQ(nn.Module):

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        nbits_w=32, nbits_a=32,
        adc_list=None,
        conv_block: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(InceptionAuxQ, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2dQ
        self.conv0 = conv_block(in_channels, 128, kernel_size=1, nbits_w=nbits_w, nbits_a=nbits_a, adc_list=adc_list)
        self.conv1 = conv_block(128, 768, kernel_size=5, nbits_w=nbits_w, nbits_a=nbits_a, adc_list=adc_list)
        self.conv1.stddev = 0.01  # type: ignore[assignment]
        #self.fc = nn.Linear(768, num_classes)
        self.fc = Linear_Q(768, num_classes, bias=True, bit=nbits_w)
        self.fc.stddev = 0.001  # type: ignore[assignment]

    def avg_pool(self, inp_dict):
        oup = F.avg_pool2d(inp_dict['act'], kernel_size=5, stride=3)
        return {'act': oup, 'alpha': inp_dict['alpha']}
    
    def adaptive_avg_pool(self, inp_dict):
        oup = F.adaptive_avg_pool2d(inp_dict['act'], (1, 1))
        return {'act': oup, 'alpha': inp_dict['alpha']}

    def forward(self, x_dict: dict) -> dict:
        # N x 768 x 17 x 17
        #x = F.avg_pool2d(x, kernel_size=5, stride=3)
        x = self.avg_pool(x_dict)
        # N x 768 x 5 x 5
        x = self.conv0(x)
        # N x 128 x 5 x 5
        x = self.conv1(x)
        # N x 768 x 1 x 1
        # Adaptive average pooling
        #x = F.adaptive_avg_pool2d(x, (1, 1))
        x = self.adaptive_avg_pool(x)['act']
        # N x 768 x 1 x 1
        x = torch.flatten(x, 1)
        # N x 768
        x = self.fc(x)
        # N x 1000
        return x


class BasicConv2dQ_first_layer(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        nbits_w=32, nbits_a=32,
        **kwargs: Any
    ) -> None:
        super(BasicConv2dQ_first_layer, self).__init__()
        #self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.conv = Conv2d_quan(in_channels, out_channels, bias=False, bit=nbits_w, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)
        self.relu = nn.ReLU(inplace=True)
        self.act = ACT_Q(bit=nbits_a)

    def forward(self, x: Tensor) -> dict:
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.act(x)
        return x


class BasicConv2dQ(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        nbits_w=32, nbits_a=32,
        adc_list=None,
        **kwargs: Any
    ) -> None:
        super(BasicConv2dQ, self).__init__()
        #self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.conv = Conv2d_quan_cim(in_channels, out_channels, bias=False, bit=nbits_w, adc_list=adc_list, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)
        self.relu = nn.ReLU(inplace=True)
        self.act = ACT_Q(bit=nbits_a)

    def forward(self, x: dict) -> dict:
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.act(x)
        return x
