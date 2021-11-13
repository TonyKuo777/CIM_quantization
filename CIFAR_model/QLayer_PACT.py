import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

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
