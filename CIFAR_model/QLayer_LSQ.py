import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from enum import Enum
from torch.nn.parameter import Parameter
#from models.modules import _Conv2dQ, Qmodes, _LinearQ, _ActQ

import ipdb

class Qmodes(Enum):
    layer_wise = 1
    kernel_wise = 2


class _Conv2dQ(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, bitW=32, bitA=32, bitO=32, sub_channel='v', dilation=1, groups=1, bias=True, **kwargs_q):
        super(_Conv2dQ, self).__init__(in_channels, out_channels, kernel_size, stride=stride,
                                       padding=padding, bitW=bitW, bitA=bitA, bitO=bitO, sub_channel=sub_channel, dilation=dilation, groups=groups, bias=bias)
        self.kwargs_q = get_default_kwargs_q(kwargs_q, layer_type=self)
        self.bitW = kwargs_q['bitW']
        if self.bitW < 0:
            self.register_parameter('alpha', None)
            return
        self.q_mode = kwargs_q['mode']
        if self.q_mode == Qmodes.kernel_wise:
            self.alpha = Parameter(torch.Tensor(out_channels))
        else:  # layer-wise quantization
            self.alpha = Parameter(torch.Tensor(1))
        self.register_buffer('init_state', torch.zeros(1))

    def add_param(self, param_k, param_v):
        self.kwargs_q[param_k] = param_v

    def extra_repr(self):
        s_prefix = super(_Conv2dQ, self).extra_repr()
        if self.alpha is None:
            return '{}, fake'.format(s_prefix)
        return '{}, {}'.format(s_prefix, self.kwargs_q)

class _LinearQ(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, **kwargs_q):                               
        super(_LinearQ, self).__init__(in_features=in_features, out_features=out_features, bias=bias)
        self.kwargs_q = get_default_kwargs_q(kwargs_q, layer_type=self)
        self.nbits = kwargs_q['nbits']
        if self.nbits < 0:
            self.register_parameter('alpha', None)
            return
        self.alpha = Parameter(torch.Tensor(1))
        self.register_buffer('init_state', torch.zeros(1))

    def add_param(self, param_k, param_v):
        self.kwargs_q[param_k] = param_v

    def extra_repr(self):
        s_prefix = super(_LinearQ, self).extra_repr()
        if self.alpha is None:
            return '{}, fake'.format(s_prefix)
        return '{}, {}'.format(s_prefix, self.kwargs_q)



class _ActQ(nn.Module):
    def __init__(self, **kwargs_q):
        super(_ActQ, self).__init__()
        self.kwargs_q = get_default_kwargs_q(kwargs_q, layer_type=self)
        self.bitA = kwargs_q['bitA']
        if self.bitA < 0:
            self.register_parameter('alpha', None)
            return
        self.signed = kwargs_q['signed']
        self.alpha = Parameter(torch.Tensor(1))
        self.register_buffer('init_state', torch.zeros(1))

    def add_param(self, param_k, param_v):
        self.kwargs_q[param_k] = param_v

    def extra_repr(self):
        # s_prefix = super(_ActQ, self).extra_repr()
        if self.alpha is None:
            return 'fake'
        return '{}'.format(self.kwargs_q)




__all__ = ['Conv2dLSQ', 'LinearLSQ', 'ActLSQ']


class FunLSQ(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight, alpha, g, Qn, Qp):
        assert alpha > 0, 'alpha = {}'.format(alpha)
        ctx.save_for_backward(weight, alpha)
        ctx.other = g, Qn, Qp
        q_w = (weight / alpha).round().clamp(Qn, Qp)
        w_q = q_w * alpha
        return w_q

    @staticmethod
    def backward(ctx, grad_weight):
        weight, alpha = ctx.saved_tensors
        g, Qn, Qp = ctx.other
        q_w = weight / alpha
        indicate_small = (q_w < Qn).float()
        indicate_big = (q_w > Qp).float()
        indicate_middle = torch.ones(indicate_small.shape).to(indicate_small.device) - indicate_small - indicate_big
        grad_alpha = ((indicate_small * Qn + indicate_big * Qp + indicate_middle * (
                -q_w + q_w.round())) * grad_weight * g).sum().unsqueeze(dim=0)
        grad_weight = indicate_middle * grad_weight
        return grad_weight, grad_alpha, None, None, None


def grad_scale(x, scale):
    y = x
    y_grad = x * scale
    return y.detach() - y_grad.detach() + y_grad


def round_pass(x):
    y = x.round()
    y_grad = x
    return y.detach() - y_grad.detach() + y_grad

'''
class Conv2d_SRAM(nn.Conv2d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size, 
                 stride=1,
                 padding=0,
                 bitW=32,
                 bitA=32,
                 bitO=32,
                 sub_channel='v',
                 dilation=1, 
                 groups=1, 
                 bias=False):
        super(Conv2d_SRAM, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.bitW = bitW
        self.bitO = bitO
        
        self.sub_flag = sub_channel == 'v' or self.in_channels==1 or self.in_channels==3
        self.fw = fw(bitW)
        self.macs = self.weight.shape[1:].numel()
        self.fo = fo(bitO)
        
        
    def forward(self, input, order=None):
        q_weight = self.fw(self.weight)
        #q_weight = self.fw(self.weight, self.moving_var, 0.0, self.gamma, self.beta)
        
        conv_sub_layers = F.conv2d(input, q_weight, self.bias, self.stride,
                                        self.padding, self.dilation, self.groups)
        outputs = self.fo(conv_sub_layers)
        return outputs
'''


class Conv2dLSQ(_Conv2dQ):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, bitW=32, bitA=32, bitO=32, sub_channel='v', dilation=1, groups=1, bias=True,
                 mode=Qmodes.layer_wise):
        super(Conv2dLSQ, self).__init__(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, bitW=bitW, bitA=bitA, bitO=bitO, sub_channel=sub_channel, dilation=dilation, groups=groups, bias=bias, mode=mode)

    def forward(self, x):
        if self.alpha is None:
            return F.conv2d(x, self.weight, self.bias, self.stride,
                            self.padding, self.dilation, self.groups)
        Qn = -2 ** (self.bitW - 1)
        Qp = 2 ** (self.bitW - 1) - 1
        if self.training and self.init_state == 0:
            self.alpha.data.copy_(2 * self.weight.abs().mean() / math.sqrt(Qp))
            self.init_state.fill_(1)
        
        g = 1.0 / math.sqrt(self.weight.numel() * Qp)

        # Method1: 31GB GPU memory (AlexNet w4a4 bs 2048) 17min/epoch
        alpha = grad_scale(self.alpha, g)
        w_q = round_pass((self.weight / alpha).clamp(Qn, Qp)) * alpha

        # Method2: 25GB GPU memory (AlexNet w4a4 bs 2048) 32min/epoch
        # w_q = FunLSQ.apply(self.weight, self.alpha, g, Qn, Qp)
        return F.conv2d(x, w_q, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


class LinearLSQ(_LinearQ):
    def __init__(self, in_features, out_features, bias=True, nbits=4):
        super(LinearLSQ, self).__init__(in_features=in_features, out_features=out_features, bias=bias, nbits=nbits)

    def forward(self, x):
        if self.alpha is None:
            return F.linear(x, self.weight, self.bias)
        Qn = -2 ** (self.nbits - 1)
        Qp = 2 ** (self.nbits - 1) - 1
        if self.training and self.init_state == 0:
            self.alpha.data.copy_(2 * self.weight.abs().mean() / math.sqrt(Qp))
            self.init_state.fill_(1)
        g = 1.0 / math.sqrt(self.weight.numel() * Qp)

        # Method1:
        alpha = grad_scale(self.alpha, g)
        w_q = round_pass((self.weight / alpha).clamp(Qn, Qp)) * alpha

        # Method2:
        # w_q = FunLSQ.apply(self.weight, self.alpha, g, Qn, Qp)
        return F.linear(x, w_q, self.bias)


class ActLSQ(_ActQ):
    def __init(self, bitA=32, signed=False):
        super(ActLSQ, self).__init(bitA=bitA, signed=signed)

    def forward(self, x):
        if self.alpha is None:
            return x
        if self.signed:
            Qn = -2 ** (self.bitA - 1)
            Qp = 2 ** (self.bitA - 1) - 1
        else:
            Qn = 0
            Qp = 2 ** self.bitA - 1
        if self.training and self.init_state == 0:
            self.alpha.data.copy_(2 * x.abs().mean() / math.sqrt(Qp))
            self.init_state.fill_(1)

        g = 1.0 / math.sqrt(x.numel() * Qp)

        # Method1:
        alpha = grad_scale(self.alpha, g)
        x = round_pass((x / alpha).clamp(Qn, Qp)) * alpha

        # Method2:
        # x_q = FunLSQ.apply(x, self.alpha, g, Qn, Qp)
        return x















'''
class quantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, k, clip_val=1.0):
        if k == 32:
            return input
        elif k == 1:
            output = torch.sign(input)
        else:
            n = float(2 ** k - 1)
            scale = n / clip_val
            output = torch.round(input * scale) / scale
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input, None, None

class fw(nn.Module):
    def __init__(self, bitW, mode=None):
        super(fw, self).__init__()
        self.bitW = bitW
        self.quantize = quantize().apply
        self.mode = mode
        #self.bn = BNW()
    
    def forward(self, x):
        if self.bitW == 32:
            return x

        elif self.bitW == 1:
            E = torch.mean(torch.abs(x)).detach()
            qx = self.quantize(x / E, self.bitW) * E
            return qx

        elif self.mode == 'ReRam':
            tanh_x = x.tanh()
            max_x = tanh_x.abs().max()
            qx = tanh_x / max_x
            qx = self.quantize(qx, self.bitW-1) #* max_x
            return qx

        else:
            tanh_x = x.tanh()
            max_x = tanh_x.abs().max()
            qx = tanh_x / max_x    
            qx = qx * 0.5 + 0.5
            qx = (2.0 * self.quantize(qx, self.bitW) - 1.0) * max_x
            
            return qx

class fo(nn.Module):
    def __init__(self, bitO):
        super(fo, self).__init__()
        self.bitO = bitO
        self.quantize = quantize().apply
    
    def forward(self, x):
        if self.bitO == 32:
            return x

        else:
            n = 2**(self.bitO-1)
            x = x.clamp(-n, n) / n
            return self.quantize(x, self.bitO-1)
        
class Conv2d_Q(nn.Conv2d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size, 
                 stride=1,
                 padding=0,
                 bitW=32,
                 dilation=1, 
                 groups=1, 
                 bias=False):
        super(Conv2d_Q, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.bitW = bitW
        self.fw = fw(bitW)

    def forward(self, input, order=None):
        q_weight = self.fw(self.weight)

        outputs = F.conv2d(input, q_weight, self.bias, self.stride,
                           self.padding, self.dilation, self.groups)

        return outputs 



class Act_Q(nn.Module):
    def __init__(self, bitA, boundary=1.0):
        super(Act_Q, self).__init__()
        self.bitA = bitA
        self.clip_val = nn.Parameter(torch.Tensor([boundary]))
        self.quantize = quantize().apply
    
    def forward(self, x):
        if self.bitA==32 or self.bitA is None:
            # max(x, 0.0)
            qa = torch.nn.functional.relu(x)
        else:
            qa = torch.clamp(x, min=0)
            #print(self.clip_val.shape)
            qa = torch.where(qa < self.clip_val, qa, self.clip_val)
            qa = self.quantize(qa, self.bitA, self.clip_val.detach())
            return qa
    
class Act6_Q(nn.Module):
    def __init__(self, bitA=32):
        super(Act6_Q, self).__init__()
        self.bitA = bitA
        self.quantize = quantize().apply
    
    def forward(self, x):
        if self.bitA==32:
            # max(x, 0.0)
            qa = torch.nn.functional.relu6(x)
        else:
            # min(max(x, 0), 1)
            qa = self.quantize(torch.clamp(x, 0.0, 6.0), self.bitA)
        return qa

class Linear_Q(nn.Linear):
    def __init__(self,
                 in_features, 
                 out_features, 
                 bitW=32,
                 bias=True):
        super(Linear_Q, self).__init__(in_features, out_features, bias)
        self.bitW = bitW
        self.fw = fw(bitW)

    def forward(self, input):
        q_weight = self.fw(self.weight)
        return F.linear(input, q_weight, self.bias)


class Conv2d_R2Q(nn.Conv2d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size, 
                 stride=1,
                 padding=0,
                 bitW=32,
                 bitO=32,
                 sub_channel='v',
                 mode='ReRam',
                 dilation=1, 
                 groups=1, 
                 bias=False):
        super(Conv2d_R2Q, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.bitW = bitW
        self.bitO = bitO
        self.fw = fw(bitW, mode)
        self.fo = fo(bitO)
        self.sub_channel = sub_channel

    def forward(self, input, order=None):
        q_weight = self.fw(self.weight)

        if self.sub_channel=='v' or self.in_channels==1 or self.in_channels==3:
            conv_sub_layers = F.conv2d(input, q_weight, self.bias, self.stride,
                                        self.padding, self.dilation, self.groups)
            outputs = self.fo(conv_sub_layers)
            return outputs

        else:
            # input [N, CH, H, W]
            #in_N, in_CH, in_H, in_W = input.shape

            #groups = self.in_channels // self.sub_channel
            input_slice = input.split(self.sub_channel, 1)#input.reshape(-1, groups, self.sub_channel, in_H, in_W)
            qw_slice = q_weight.split(self.sub_channel, 1)#q_weight.reshape(self.out_channels, groups, self)

            conv_sub_layers = torch.stack([F.conv2d(i, f, self.bias, self.stride, self.padding, self.dilation, self.groups)\
                 for i, f in zip(input_slice, qw_slice)], axis=4)

            SA_sub_out = self.fo(conv_sub_layers)
            outputs = SA_sub_out.sum(4)
            return outputs

class Conv2d_SRAM(nn.Conv2d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size, 
                 stride=1,
                 padding=0,
                 bitW=32,
                 bitA=32,
                 bitO=32,
                 sub_channel='v',
                 dilation=1, 
                 groups=1, 
                 bias=False):
        super(Conv2d_SRAM, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.bitW = bitW
        self.bitO = bitO
        
        self.sub_flag = sub_channel == 'v' or self.in_channels==1 or self.in_channels==3
        self.fw = fw(bitW)
        self.macs = self.weight.shape[1:].numel()
        self.fo = fo(bitO)
        
        
    def forward(self, input, order=None):
        q_weight = self.fw(self.weight)
        #q_weight = self.fw(self.weight, self.moving_var, 0.0, self.gamma, self.beta)
        
        conv_sub_layers = F.conv2d(input, q_weight, self.bias, self.stride,
                                        self.padding, self.dilation, self.groups)
        outputs = self.fo(conv_sub_layers)
        return outputs
'''