import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.modules.utils import _pair
from torch.nn.parameter import Parameter
from torch.nn import init
import math
import os

class LinearQuant(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, bits, scale):
        n = 2 ** bits - 1
        return torch.round(x / scale * n) / n * scale

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None

class PactTransform(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, clip_val):
        ctx.save_for_backward(x, clip_val)
        return torch.clamp(x, 0.0, clip_val.data[0])
    
    @staticmethod
    def backward(ctx, grad_output):
        x, clip_val = ctx.saved_tensors
        
        grad_input = grad_output.clone()
        grad_input.masked_fill_(x.le(0), 0)
        grad_input.masked_fill_(x.ge(clip_val.data[0]), 0)

        grad_clip_val = grad_output.clone()
        grad_clip_val.masked_fill_(x.lt(clip_val.data[0]), 0)
        grad_clip_val = grad_clip_val.sum().expand_as(clip_val)
        
        return grad_input, grad_clip_val      

class Act_Q(nn.Module):
    def __init__(self, bitA):
        super(Act_Q, self).__init__()
        self.bits = bitA
        self.transformer = PactTransform.apply
        self.discritizer = LinearQuant.apply

        self.clip_val = nn.Parameter(torch.Tensor([1.0])) # initial value set to 1.0, user cab try other value.

    def forward(self, x):
        x = self.transformer(x, self.clip_val)
        x = self.discritizer(x, self.bits, self.clip_val)
        x_dict = {'act': x, 'alpha': self.clip_val / (2 ** self.bits - 1)}
        return x_dict

class quantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, k):
        if k == 32:
            return input
        elif k == 1:
            output = torch.sign(input)
        else:
            n = float(2 ** k - 1)
            output = torch.round(input * n) / n
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input, None    #, None

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
        
        else:
            tanh_x = x.tanh()
            max_x = tanh_x.abs().max()
            qx = tanh_x / max_x
            qx = self.quantize(qx, self.bitW-1) #* max_x
            return qx
        '''
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
        '''

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

class sram_cim_conv(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, stride, padding, scale_alpha, scale_alpha_w, adc_list, bias=None):
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
        ctx.save_for_backward(input, weight, bias)
        ctx.stride = stride
        ctx.padding = padding 

        B, CH, H, W = input.shape
        K, CH, KH, KW = weight.shape
        alpha = 16                                                  # alpha is the MAC number in CIM

        # 16 channel as a Group
        GPS = CH//alpha                                             # GPS means GrouPS
        input_slices = input.view(B, GPS, alpha, H, W)              # [B, CH//alpha, 16, H, W]
        weight_slices = weight.view(K, GPS, alpha, KH, KW)          # [K, CH//alpha, 16, KH, KW]

        # Initialize the OFM
        # calculate output height and width
        OH = int( (H - KH + 2 * padding) / stride + 1 )
        OW = int( (W - KW + 2 * padding) / stride + 1 )
        output = torch.zeros((B, K, OH, OW))                                        # [B, K, OH, OW]
        
        for gp in range(GPS):
            input_unfold = torch.nn.functional.unfold(input_slices[:, gp, :, :, :], # [B, alpha*KH*KW, OH*OW]
                                                    kernel_size=(KH, KW), 
                                                    stride=stride, 
                                                    padding=padding)
            input_unfold = input_unfold.transpose(1, 2)                             # [B, OH*OW, alpha*KH*KW]
            input_unfold = input_unfold.view(B, OH*OW, KH*KW, alpha)                # [B, OH*OW, KH*KW, alpha]
            
            weight_unfold = weight_slices[:, gp, :, :, :].view(K, -1).t()           # [alpha*KH*KW, K]
            weight_unfold = weight_unfold.view(KH*KW, alpha, K)                     # [KH*KW, alpha, K]
                
            output_unfold = torch.zeros((B, OH*OW, K))                              # [B, OH*OW, K]
            
            for i in range(KH*KW):
                # 8a8w
                # FP --> Int
                x_int = torch.round(input_unfold[:, :, i, :] / scale_alpha).int()   # 255 <- 2**8 - 1
                w_int = torch.round(weight_unfold[i, :, :] * scale_alpha_w).int()   # [-127, 127]
                
                msb_x_int = x_int >> 4
                lsb_x_int = x_int & 15
                
                output_unfold += ( (CIM_MAC(msb_x_int, w_int, adc_list) << 4) + CIM_MAC(lsb_x_int, w_int, adc_list) )
                
            output_unfold = output_unfold.transpose(1, 2)                            # [B, K, OH*OW]
            output += torch.nn.functional.fold(output_unfold, (OH, OW), (1, 1))
        output = output * scale_alpha / scale_alpha_w   # 8a8w

        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_variables
        stride = ctx.stride
        padding = ctx.padding
            
        grad_input = grad_weight = grad_bias = None

        if ctx.needs_input_grad[0]:
            grad_input = torch.nn.grad.conv2d_input(input.shape, weight, grad_output, stride, padding)
            
        if ctx.needs_input_grad[1]:
            grad_weight = torch.nn.grad.conv2d_weight(input, weight.shape, grad_output, stride, padding)
        
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = None
        if bias is not None:
            return grad_input, grad_weight, None, None, grad_bias
    
        return grad_input, grad_weight, None, None, None, None, None, None

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

class Linear_Q(nn.Linear):
    def __init__(self,
                 in_features, 
                 out_features, 
                 bitW=32,
                 bias=False):
        super(Linear_Q, self).__init__(in_features, out_features, bias)
        self.bitW = bitW
        self.fw = fw(bitW)

    def forward(self, input):
        q_weight = self.fw(self.weight)
        return F.linear(input, q_weight, self.bias)

class Conv2d_SRAM(nn.Conv2d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size, 
                 stride=1,
                 padding=0,
                 bitW=32,
                 #bitA=32,
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

def w_nonideal(x):
    map = torch.rand(x.shape)
    temp = x #.cpu()
    '''
    # approximate
    x[(temp >= 2) & (temp <= 6) & (map < 0.01)] -= 1
    x[(temp >= 2) & (temp <= 6) & (map >= 0.98)] += 1
    
    x[(temp >= 7) & (temp <= 14) & (map < 0.005)] -= 2
    x[(temp >= 7) & (temp <= 14) & (map >= 0.005) & (map < 0.025)] -= 1
    x[(temp >= 7) & (temp <= 14) & (map >= 0.975) & (map < 0.995)] += 1
    x[(temp >= 7) & (temp <= 14) & (map >= 0.995)] += 2
    
    x[(temp >= 15) & (temp <= 31) & (map < 0.01)] -= 2
    x[(temp >= 15) & (temp <= 31) & (map >= 0.01) & (map < 0.08)] -= 1
    x[(temp >= 15) & (temp <= 31) & (map >= 0.92) & (map < 0.99)] += 1
    x[(temp >= 15) & (temp <= 31) & (map >= 0.99)] += 2
    
    x = torch.clamp(x, 0, 31)
    '''
    x = x.masked_fill((temp == 1) & (map < 0.003), 0.)
    x = x.masked_fill((temp == 1) & (map >= 0.999), 2.)
    
    x = x.masked_fill((temp == 2) & (map < 0.001), 1.)
    x = x.masked_fill((temp == 2) & (map >= 0.962) & (map < 0.986), 3.)
    x = x.masked_fill((temp == 2) & (map >= 0.986), 4.)
    
    x = x.masked_fill((temp == 3) & (map < 0.006), 2.)
    x = x.masked_fill((temp == 3) & (map >= 0.978) & (map < 0.993), 4.)
    x = x.masked_fill((temp == 3) & (map >= 0.993), 5.)
    
    x = x.masked_fill((temp == 4) & (map < 0.008), 3.)
    x = x.masked_fill((temp == 4) & (map >= 0.989) & (map < 0.997), 5.)
    x = x.masked_fill((temp == 4) & (map >= 0.997), 6.)
    
    x = x.masked_fill((temp == 5) & (map < 0.004), 3.)
    x = x.masked_fill((temp == 5) & (map >= 0.004) & (map < 0.015), 4.)
    x = x.masked_fill((temp == 5) & (map >= 0.984) & (map < 0.996), 6.)
    x = x.masked_fill((temp == 5) & (map >= 0.996), 7.)
    
    x = x.masked_fill((temp == 6) & (map < 0.014), 5.)
    x = x.masked_fill((temp == 6) & (map >= 0.986) & (map < 0.997), 7.)
    x = x.masked_fill((temp == 6) & (map >= 0.997), 8.)
    
    x = x.masked_fill((temp == 7) & (map < 0.004), 5.)
    x = x.masked_fill((temp == 7) & (map >= 0.004) & (map < 0.021), 6.)
    x = x.masked_fill((temp == 7) & (map >= 0.975) & (map < 0.995), 8.)
    x = x.masked_fill((temp == 7) & (map >= 0.995), 9.)

    x = x.masked_fill((temp == 8) & (map < 0.01), 6.)
    x = x.masked_fill((temp == 8) & (map >= 0.01) & (map < 0.029), 7.)
    x = x.masked_fill((temp == 8) & (map >= 0.987) & (map < 0.999), 9.)
    x = x.masked_fill((temp == 8) & (map >= 0.999), 10.)

    x = x.masked_fill((temp == 9) & (map < 0.005), 7.)
    x = x.masked_fill((temp == 9) & (map >= 0.005) & (map < 0.06), 8.)
    x = x.masked_fill((temp == 9) & (map >= 0.973) & (map < 0.995), 10.)
    x = x.masked_fill((temp == 9) & (map >= 0.995), 11.)

    x = x.masked_fill((temp == 10) & (map < 0.005), 8.)
    x = x.masked_fill((temp == 10) & (map >= 0.005) & (map < 0.054), 9.)
    x = x.masked_fill((temp == 10) & (map >= 0.975) & (map < 0.996), 11.)
    x = x.masked_fill((temp == 10) & (map >= 0.996), 12.)

    x = x.masked_fill((temp == 11) & (map < 0.008), 9.)
    x = x.masked_fill((temp == 11) & (map >= 0.008) & (map < 0.041), 10.)
    x = x.masked_fill((temp == 11) & (map >= 0.973) & (map < 0.995), 12.)
    x = x.masked_fill((temp == 11) & (map >= 0.995), 13.)

    x = x.masked_fill((temp == 12) & (map < 0.01), 10.)
    x = x.masked_fill((temp == 12) & (map >= 0.01) & (map < 0.035), 11.)
    x = x.masked_fill((temp == 12) & (map >= 0.987) & (map < 0.996), 13.)
    x = x.masked_fill((temp == 12) & (map >= 0.996), 14.)

    x = x.masked_fill((temp == 13) & (map < 0.013), 12.)
    x = x.masked_fill((temp == 13) & (map >= 0.986), 14.)

    x = x.masked_fill((temp == 14) & (map < 0.011), 12.)
    x = x.masked_fill((temp == 14) & (map >= 0.011) & (map < 0.025), 13.)
    x = x.masked_fill((temp == 14) & (map >= 0.986) & (map < 0.997), 15.)
    x = x.masked_fill((temp == 14) & (map >= 0.997), 16.)

    x = x.masked_fill((temp == 15) & (map < 0.008), 13.)
    x = x.masked_fill((temp == 15) & (map >= 0.008) & (map < 0.05), 14.)
    x = x.masked_fill((temp == 15) & (map >= 0.982) & (map < 0.999), 16.)
    x = x.masked_fill((temp == 15) & (map >= 0.999), 17.)

    x = x.masked_fill((temp == 16) & (map < 0.013), 14.)
    x = x.masked_fill((temp == 16) & (map >= 0.013) & (map < 0.077), 15.)
    x = x.masked_fill((temp == 16) & (map >= 0.928) & (map < 0.986), 17.)
    x = x.masked_fill((temp == 16) & (map >= 0.986), 18.)

    x = x.masked_fill((temp == 17) & (map < 0.026), 15.)
    x = x.masked_fill((temp == 17) & (map >= 0.026) & (map < 0.114), 16.)
    x = x.masked_fill((temp == 17) & (map >= 0.935) & (map < 0.979), 18.)
    x = x.masked_fill((temp == 17) & (map >= 0.979), 19.)

    x = x.masked_fill((temp == 18) & (map < 0.06), 15.)
    x = x.masked_fill((temp == 18) & (map >= 0.06) & (map < 0.03), 16.)
    x = x.masked_fill((temp == 18) & (map >= 0.03) & (map < 0.156), 17.)
    x = x.masked_fill((temp == 18) & (map >= 0.914) & (map < 0.987), 19.)
    x = x.masked_fill((temp == 18) & (map >= 0.987), 20.)

    x = x.masked_fill((temp == 19) & (map < 0.005), 16.)
    x = x.masked_fill((temp == 19) & (map >= 0.005) & (map < 0.022), 17.)
    x = x.masked_fill((temp == 19) & (map >= 0.022) & (map < 0.153), 18.)
    x = x.masked_fill((temp == 19) & (map >= 0.874) & (map < 0.976), 20.)
    x = x.masked_fill((temp == 19) & (map >= 0.976) & (map < 0.995), 21.)
    x = x.masked_fill((temp == 19) & (map >= 0.995), 22.)

    x = x.masked_fill((temp == 20) & (map < 0.01), 17.)
    x = x.masked_fill((temp == 20) & (map >= 0.01) & (map < 0.031), 18.)
    x = x.masked_fill((temp == 20) & (map >= 0.031) & (map < 0.157), 19.)
    x = x.masked_fill((temp == 20) & (map >= 0.885) & (map < 0.994), 21.)
    x = x.masked_fill((temp == 20) & (map >= 0.994) & (map < 0.999), 22.)
    x = x.masked_fill((temp == 20) & (map >= 0.999), 23.)

    x = x.masked_fill((temp == 21) & (map < 0.006), 18.)
    x = x.masked_fill((temp == 21) & (map >= 0.006) & (map < 0.019), 19.)
    x = x.masked_fill((temp == 21) & (map >= 0.019) & (map < 0.133), 20.)
    x = x.masked_fill((temp == 21) & (map >= 0.868) & (map < 0.986), 22.)
    x = x.masked_fill((temp == 21) & (map >= 0.986) & (map < 0.998), 23.)
    x = x.masked_fill((temp == 21) & (map >= 0.998), 24.)

    x = x.masked_fill((temp == 22) & (map < 0.002), 19.)
    x = x.masked_fill((temp == 22) & (map >= 0.002) & (map < 0.033), 20.)
    x = x.masked_fill((temp == 22) & (map >= 0.033) & (map < 0.156), 21.)
    x = x.masked_fill((temp == 22) & (map >= 0.899) & (map < 0.971), 23.)
    x = x.masked_fill((temp == 22) & (map >= 0.971) & (map < 0.996), 24.)
    x = x.masked_fill((temp == 22) & (map >= 0.996), 25.)

    x = x.masked_fill((temp == 23) & (map < 0.002), 20.)
    x = x.masked_fill((temp == 23) & (map >= 0.002) & (map < 0.025), 21.)
    x = x.masked_fill((temp == 23) & (map >= 0.025) & (map < 0.157), 22.)
    x = x.masked_fill((temp == 23) & (map >= 0.936) & (map < 0.993), 24.)
    x = x.masked_fill((temp == 23) & (map >= 0.993), 25.)

    x = x.masked_fill((temp == 24) & (map < 0.002), 21.)
    x = x.masked_fill((temp == 24) & (map >= 0.002) & (map < 0.023), 22.)
    x = x.masked_fill((temp == 24) & (map >= 0.023) & (map < 0.104), 23.)
    x = x.masked_fill((temp == 24) & (map >= 0.916) & (map < 0.99), 25.)
    x = x.masked_fill((temp == 24) & (map >= 0.99), 26.)

    x = x.masked_fill((temp == 25) & (map < 0.003), 22.)
    x = x.masked_fill((temp == 25) & (map >= 0.003) & (map < 0.011), 23.)
    x = x.masked_fill((temp == 25) & (map >= 0.011) & (map < 0.136), 24.)
    x = x.masked_fill((temp == 25) & (map >= 0.904) & (map < 0.992), 26.)
    x = x.masked_fill((temp == 25) & (map >= 0.992), 27.)

    x = x.masked_fill((temp == 26) & (map < 0.006), 23.)
    x = x.masked_fill((temp == 26) & (map >= 0.006) & (map < 0.014), 24.)
    x = x.masked_fill((temp == 26) & (map >= 0.014) & (map < 0.149), 25.)
    x = x.masked_fill((temp == 26) & (map >= 0.955) & (map < 0.994), 27.)
    x = x.masked_fill((temp == 26) & (map >= 0.994), 28.)

    x = x.masked_fill((temp == 27) & (map < 0.001), 25.)
    x = x.masked_fill((temp == 27) & (map >= 0.001) & (map < 0.013), 26.)
    x = x.masked_fill((temp == 27) & (map >= 0.974) & (map < 0.995), 28.)
    x = x.masked_fill((temp == 27) & (map >= 0.995), 29.)

    x = x.masked_fill((temp == 28) & (map < 0.002), 26.)
    x = x.masked_fill((temp == 28) & (map >= 0.002) & (map < 0.015), 27.)
    x = x.masked_fill((temp == 28) & (map >= 0.986) & (map < 0.998), 29.)
    x = x.masked_fill((temp == 28) & (map >= 0.998), 30.)

    x = x.masked_fill((temp == 29) & (map < 0.003), 27.)
    x = x.masked_fill((temp == 29) & (map >= 0.003) & (map < 0.018), 28.)
    x = x.masked_fill((temp == 29) & (map >= 0.996), 30.)

    x = x.masked_fill((temp == 30) & (map < 0.014), 29.)
    x = x.masked_fill((temp == 30) & (map >= 0.989), 31.)

    x = x.masked_fill((temp == 31) & (map < 0.001), 29.)
    x = x.masked_fill((temp == 31) & (map >= 0.001) & (map < 0.009), 30.)
    
    ########################################
    # (ImNet-Res50) Revise non ideal model #
    ########################################
    #map = torch.rand(x.shape)
    temp_ni = x#.cpu()
    
    x = x.masked_fill((temp_ni == 0) & (map < 0.001), 1.)

    x = x.masked_fill((temp_ni == 1) & (map < 0.001), 2.)

    x = x.masked_fill((temp_ni == 2) & (map < 0.006), 3.)
    x = x.masked_fill((temp_ni == 2) & (map >= 0.999), 1.)
    
    x = x.masked_fill((temp_ni == 3) & (map < 0.007), 5.)
    x = x.masked_fill((temp_ni == 3) & (map >= 0.007) & (map < 0.018), 4.)
    x = x.masked_fill((temp_ni == 3) & (map >= 0.972), 2.)
    
    x = x.masked_fill((temp_ni == 4) & (map < 0.017), 5.)
    x = x.masked_fill((temp_ni == 4) & (map >= 0.959) & (map < 0.98), 3.)
    x = x.masked_fill((temp_ni == 4) & (map >= 0.98), 2.)

    x = x.masked_fill((temp_ni == 5) & (map < 0.004), 7.)
    x = x.masked_fill((temp_ni == 5) & (map >= 0.004) & (map < 0.02), 6.)
    x = x.masked_fill((temp_ni == 5) & (map >= 0.983) & (map < 0.993), 4.)
    x = x.masked_fill((temp_ni == 5) & (map >= 0.993), 3.)

    x = x.masked_fill((temp_ni == 6) & (map < 0.01), 8.)
    x = x.masked_fill((temp_ni == 6) & (map >= 0.01) & (map < 0.027), 7.)
    x = x.masked_fill((temp_ni == 6) & (map >= 0.985) & (map < 0.998), 5.)
    x = x.masked_fill((temp_ni == 6) & (map >= 0.998), 4.)

    x = x.masked_fill((temp_ni == 7) & (map < 0.005), 9.)
    x = x.masked_fill((temp_ni == 7) & (map >= 0.005) & (map < 0.023), 8.)
    x = x.masked_fill((temp_ni == 7) & (map >= 0.985) & (map < 0.996), 6.)
    x = x.masked_fill((temp_ni == 7) & (map >= 0.996), 5.)

    x = x.masked_fill((temp_ni == 8) & (map < 0.004), 10.)
    x = x.masked_fill((temp_ni == 8) & (map >= 0.004) & (map < 0.058), 9.)
    x = x.masked_fill((temp_ni == 8) & (map >= 0.979) & (map < 0.998), 7.)
    x = x.masked_fill((temp_ni == 8) & (map >= 0.998), 6.)

    x = x.masked_fill((temp_ni == 9) & (map < 0.007), 11.)
    x = x.masked_fill((temp_ni == 9) & (map >= 0.007) & (map < 0.057), 10.)
    x = x.masked_fill((temp_ni == 9) & (map >= 0.985) & (map < 0.996), 8.)
    x = x.masked_fill((temp_ni == 9) & (map >= 0.996), 7.)

    x = x.masked_fill((temp_ni == 10) & (map < 0.001), 12.)
    x = x.masked_fill((temp_ni == 10) & (map >= 0.001) & (map < 0.034), 11.)
    x = x.masked_fill((temp_ni == 10) & (map >= 0.977) & (map < 0.999), 9.)
    x = x.masked_fill((temp_ni == 10) & (map >= 0.999), 8.)

    x = x.masked_fill((temp_ni == 11) & (map < 0.034), 12.)
    x = x.masked_fill((temp_ni == 11) & (map >= 0.973) & (map < 0.995), 10.)
    x = x.masked_fill((temp_ni == 11) & (map >= 0.995), 9.)

    x = x.masked_fill((temp_ni == 12) & (map < 0.011), 14.)
    x = x.masked_fill((temp_ni == 12) & (map >= 0.011) & (map < 0.023), 13.)
    x = x.masked_fill((temp_ni == 12) & (map >= 0.974) & (map < 0.996), 11.)
    x = x.masked_fill((temp_ni == 12) & (map >= 0.996), 10.)

    x = x.masked_fill((temp_ni == 13) & (map < 0.007), 15.)
    x = x.masked_fill((temp_ni == 13) & (map >= 0.007) & (map < 0.021), 14.)
    x = x.masked_fill((temp_ni == 13) & (map >= 0.987) & (map < 0.996), 12.)
    x = x.masked_fill((temp_ni == 13) & (map >= 0.996), 11.)

    x = x.masked_fill((temp_ni == 14) & (map < 0.012), 16.)
    x = x.masked_fill((temp_ni == 14) & (map >= 0.012) & (map < 0.051), 15.)
    x = x.masked_fill((temp_ni == 14) & (map >= 0.989) & (map < 0.998), 13.)
    x = x.masked_fill((temp_ni == 14) & (map >= 0.998), 12.)

    x = x.masked_fill((temp_ni == 15) & (map < 0.02), 17.)
    x = x.masked_fill((temp_ni == 15) & (map >= 0.02) & (map < 0.078), 16.)
    x = x.masked_fill((temp_ni == 15) & (map >= 0.997), 14.)

    x = x.masked_fill((temp_ni == 16) & (map < 0.02), 18.)
    x = x.masked_fill((temp_ni == 16) & (map >= 0.02) & (map < 0.102), 17.)
    x = x.masked_fill((temp_ni == 16) & (map >= 0.989), 15.)

    x = x.masked_fill((temp_ni == 17) & (map < 0.002), 20.)
    x = x.masked_fill((temp_ni == 17) & (map >= 0.002) & (map < 0.013), 19.)
    x = x.masked_fill((temp_ni == 17) & (map >= 0.013) & (map < 0.136), 18.)
    x = x.masked_fill((temp_ni == 17) & (map >= 0.948), 16.)

    x = x.masked_fill((temp_ni == 18) & (map < 0.001), 21.)
    x = x.masked_fill((temp_ni == 18) & (map >= 0.001) & (map < 0.018), 20.)
    x = x.masked_fill((temp_ni == 18) & (map >= 0.018) & (map < 0.151), 19.)
    x = x.masked_fill((temp_ni == 18) & (map >= 0.949) & (map < 0.989), 17.)
    x = x.masked_fill((temp_ni == 18) & (map >= 0.989), 16.)

    x = x.masked_fill((temp_ni == 19) & (map < 0.004), 22.)
    x = x.masked_fill((temp_ni == 19) & (map >= 0.004) & (map < 0.018), 21.)
    x = x.masked_fill((temp_ni == 19) & (map >= 0.018) & (map < 0.147), 20.)
    x = x.masked_fill((temp_ni == 19) & (map >= 0.898) & (map < 0.977), 18.)
    x = x.masked_fill((temp_ni == 19) & (map >= 0.977), 17.)

    x = x.masked_fill((temp_ni == 20) & (map < 0.004), 23.)
    x = x.masked_fill((temp_ni == 20) & (map >= 0.004) & (map < 0.035), 22.)
    x = x.masked_fill((temp_ni == 20) & (map >= 0.035) & (map < 0.147), 21.)
    x = x.masked_fill((temp_ni == 20) & (map >= 0.877) & (map < 0.985), 19.)
    x = x.masked_fill((temp_ni == 20) & (map >= 0.985), 18.)

    x = x.masked_fill((temp_ni == 21) & (map < 0.004), 24.)
    x = x.masked_fill((temp_ni == 21) & (map >= 0.004) & (map < 0.032), 23.)
    x = x.masked_fill((temp_ni == 21) & (map >= 0.032) & (map < 0.152), 22.)
    x = x.masked_fill((temp_ni == 21) & (map >= 0.862) & (map < 0.976), 20.)
    x = x.masked_fill((temp_ni == 21) & (map >= 0.976), 19.)

    x = x.masked_fill((temp_ni == 22) & (map < 0.006), 25.)
    x = x.masked_fill((temp_ni == 22) & (map >= 0.006) & (map < 0.028), 24.)
    x = x.masked_fill((temp_ni == 22) & (map >= 0.028) & (map < 0.163), 23.)
    x = x.masked_fill((temp_ni == 22) & (map >= 0.872) & (map < 0.988), 21.)
    x = x.masked_fill((temp_ni == 22) & (map >= 0.988) & (map < 0.994), 20.)
    x = x.masked_fill((temp_ni == 22) & (map >= 0.994), 19.)

    x = x.masked_fill((temp_ni == 23) & (map < 0.01), 26.)
    x = x.masked_fill((temp_ni == 23) & (map >= 0.01) & (map < 0.023), 25.)
    x = x.masked_fill((temp_ni == 23) & (map >= 0.023) & (map < 0.106), 24.)
    x = x.masked_fill((temp_ni == 23) & (map >= 0.896) & (map < 0.976), 22.)
    x = x.masked_fill((temp_ni == 23) & (map >= 0.976) & (map < 0.994), 21.)
    x = x.masked_fill((temp_ni == 23) & (map >= 0.994), 20.)

    x = x.masked_fill((temp_ni == 24) & (map < 0.012), 26.)
    x = x.masked_fill((temp_ni == 24) & (map >= 0.012) & (map < 0.138), 25.)
    x = x.masked_fill((temp_ni == 24) & (map >= 0.916) & (map < 0.977), 23.)
    x = x.masked_fill((temp_ni == 24) & (map >= 0.977) & (map < 0.996), 22.)
    x = x.masked_fill((temp_ni == 24) & (map >= 0.996), 21.)

    x = x.masked_fill((temp_ni == 25) & (map < 0.008), 27.)
    x = x.masked_fill((temp_ni == 25) & (map >= 0.008) & (map < 0.15), 26.)
    x = x.masked_fill((temp_ni == 25) & (map >= 0.916) & (map < 0.989), 24.)
    x = x.masked_fill((temp_ni == 25) & (map >= 0.989) & (map < 0.996), 23.)
    x = x.masked_fill((temp_ni == 25) & (map >= 0.996), 22.)

    x = x.masked_fill((temp_ni == 26) & (map < 0.017), 28.)
    x = x.masked_fill((temp_ni == 26) & (map >= 0.017) & (map < 0.045), 27.)
    x = x.masked_fill((temp_ni == 26) & (map >= 0.872) & (map < 0.975), 25.)
    x = x.masked_fill((temp_ni == 26) & (map >= 0.975), 24.)

    x = x.masked_fill((temp_ni == 27) & (map < 0.011), 29.)
    x = x.masked_fill((temp_ni == 27) & (map >= 0.011) & (map < 0.031), 28.)
    x = x.masked_fill((temp_ni == 27) & (map >= 0.939) & (map < 0.985), 26.)
    x = x.masked_fill((temp_ni == 27) & (map >= 0.985), 25.)

    x = x.masked_fill((temp_ni == 28) & (map < 0.017), 29.)
    x = x.masked_fill((temp_ni == 28) & (map >= 0.967) & (map < 0.99), 27.)
    x = x.masked_fill((temp_ni == 28) & (map >= 0.99), 26.)

    x = x.masked_fill((temp_ni == 29) & (map < 0.002), 31.)
    x = x.masked_fill((temp_ni == 29) & (map >= 0.002) & (map < 0.02), 30.)
    x = x.masked_fill((temp_ni == 29) & (map >= 0.977) & (map < 0.992), 28.)
    x = x.masked_fill((temp_ni == 29) & (map >= 0.992), 27.)

    x = x.masked_fill((temp_ni == 30) & (map < 0.013), 31.)
    x = x.masked_fill((temp_ni == 30) & (map >= 0.986) & (map < 0.994), 29.)
    x = x.masked_fill((temp_ni == 30) & (map >= 0.994), 28.)

    x = x.masked_fill((temp_ni == 31) & (map >= 0.992), 30.)
    
    return x

def wo_nonideal(x):
    map = torch.rand(x.shape)
    temp = x #.cpu()
    '''
    # approximate
    x[(temp >= 2) & (temp <= 6) & (map >= 0.99)] += 1
    
    x[(temp >= 7) & (temp <= 14) & (map < 0.03)] -= 1
    x[(temp >= 7) & (temp <= 14) & (map >= 0.99)] += 1
    
    x[(temp >= 15) & (temp <= 31) & (map < 0.09)] -= 1
    x[(temp >= 15) & (temp <= 31) & (map >= 0.94)] += 1

    x = torch.clamp(x, 0, 31)
    '''
    x = x.masked_fill((temp == 1) & (map < 0.003), 0.)
    x = x.masked_fill((temp == 1) & (map >= 0.999), 2.)
    
    x = x.masked_fill((temp == 2) & (map < 0.001), 1.)
    x = x.masked_fill((temp == 2) & (map >= 0.976), 3.)
    
    x = x.masked_fill((temp == 3) & (map < 0.006), 2.)
    x = x.masked_fill((temp == 3) & (map >= 0.989), 4.)
    
    x = x.masked_fill((temp == 4) & (map < 0.008), 3.)
    x = x.masked_fill((temp == 4) & (map >= 0.999), 5.)
    
    x = x.masked_fill((temp == 5) & (map < 0.011), 4.)
    x = x.masked_fill((temp == 5) & (map >= 0.996), 6.)
    
    x = x.masked_fill((temp == 6) & (map < 0.004), 5.)
    x = x.masked_fill((temp == 6) & (map >= 0.995), 7.)
    
    x = x.masked_fill((temp == 7) & (map < 0.031), 6.)
    x = x.masked_fill((temp == 7) & (map >= 0.998), 8.)

    x = x.masked_fill((temp == 8) & (map < 0.029), 7.)
    x = x.masked_fill((temp == 8) & (map >= 0.998), 9.)

    x = x.masked_fill((temp == 9) & (map < 0.055), 8.)
    x = x.masked_fill((temp == 9) & (map >= 0.978), 10.)

    x = x.masked_fill((temp == 10) & (map < 0.063), 9.)
    x = x.masked_fill((temp == 10) & (map >= 0.998), 11.)

    x = x.masked_fill((temp == 11) & (map < 0.046), 10.)
    x = x.masked_fill((temp == 11) & (map >= 0.988), 12.)

    x = x.masked_fill((temp == 12) & (map < 0.034), 11.)
    x = x.masked_fill((temp == 12) & (map >= 0.997), 13.)

    x = x.masked_fill((temp == 13) & (map < 0.002), 12.)
    x = x.masked_fill((temp == 13) & (map >= 0.997), 14.)

    x = x.masked_fill((temp == 14) & (map < 0.025), 13.)
    x = x.masked_fill((temp == 14) & (map >= 0.997), 15.)

    x = x.masked_fill((temp == 15) & (map < 0.042), 14.)
    x = x.masked_fill((temp == 15) & (map >= 0.983), 16.)

    x = x.masked_fill((temp == 16) & (map < 0.012), 14.)
    x = x.masked_fill((temp == 16) & (map >= 0.012) & (map < 0.073), 15.)
    x = x.masked_fill((temp == 16) & (map >= 0.944) & (map < 0.986), 17.)
    x = x.masked_fill((temp == 16) & (map >= 0.986), 18.)

    x = x.masked_fill((temp == 17) & (map < 0.027), 15.)
    x = x.masked_fill((temp == 17) & (map >= 0.027) & (map < 0.115), 16.)
    x = x.masked_fill((temp == 17) & (map >= 0.936), 18.)

    x = x.masked_fill((temp == 18) & (map < 0.011), 16.)
    x = x.masked_fill((temp == 18) & (map >= 0.011) & (map < 0.155), 17.)
    x = x.masked_fill((temp == 18) & (map >= 0.913) & (map < 0.986), 19.)
    x = x.masked_fill((temp == 18) & (map >= 0.986), 20.)

    x = x.masked_fill((temp == 19) & (map < 0.007), 17.)
    x = x.masked_fill((temp == 19) & (map >= 0.007) & (map < 0.159), 18.)
    x = x.masked_fill((temp == 19) & (map >= 0.88) & (map < 0.992), 20.)
    x = x.masked_fill((temp == 19) & (map >= 0.992), 21.)

    x = x.masked_fill((temp == 20) & (map < 0.002), 18.)
    x = x.masked_fill((temp == 20) & (map >= 0.002) & (map < 0.163), 19.)
    x = x.masked_fill((temp == 20) & (map >= 0.891), 21.)

    x = x.masked_fill((temp == 21) & (map < 0.008), 19.)
    x = x.masked_fill((temp == 21) & (map >= 0.008) & (map < 0.143), 20.)
    x = x.masked_fill((temp == 21) & (map >= 0.878) & (map < 0.988), 22.)
    x = x.masked_fill((temp == 21) & (map >= 0.988), 23.)

    x = x.masked_fill((temp == 22) & (map < 0.021), 20.)
    x = x.masked_fill((temp == 22) & (map >= 0.021) & (map < 0.163), 21.)
    x = x.masked_fill((temp == 22) & (map >= 0.906) & (map < 0.987), 23.)
    x = x.masked_fill((temp == 22) & (map >= 0.987), 24.)

    x = x.masked_fill((temp == 23) & (map < 0.013), 21.)
    x = x.masked_fill((temp == 23) & (map >= 0.013) & (map < 0.164), 22.)
    x = x.masked_fill((temp == 23) & (map >= 0.943), 24.)

    x = x.masked_fill((temp == 24) & (map < 0.095), 23.)
    x = x.masked_fill((temp == 24) & (map >= 0.907), 25.)

    x = x.masked_fill((temp == 25) & (map < 0.006), 23.)
    x = x.masked_fill((temp == 25) & (map >= 0.006) & (map < 0.15), 24.)
    x = x.masked_fill((temp == 25) & (map >= 0.918), 26.)

    x = x.masked_fill((temp == 26) & (map < 0.011), 24.)
    x = x.masked_fill((temp == 26) & (map >= 0.011) & (map < 0.146), 25.)
    x = x.masked_fill((temp == 26) & (map >= 0.964), 27.)

    x = x.masked_fill((temp == 27) & (map < 0.001), 25.)
    x = x.masked_fill((temp == 27) & (map >= 0.001) & (map < 0.013), 26.)
    x = x.masked_fill((temp == 27) & (map >= 0.989), 28.)

    x = x.masked_fill((temp == 28) & (map < 0.001), 27.)
    x = x.masked_fill((temp == 28) & (map >= 0.989), 29.)

    x = x.masked_fill((temp == 29) & (map < 0.005), 28.)
    x = x.masked_fill((temp == 29) & (map >= 0.996), 30.)

    x = x.masked_fill((temp == 30) & (map < 0.011), 29.)
    x = x.masked_fill((temp == 30) & (map >= 0.996), 31.)

    x = x.masked_fill((temp == 31) & (map < 0.009), 30.)
    
    ########################################
    # (ImNet-Res50) Revise non ideal model #
    ########################################
    
    #map = torch.rand(x.shape)
    temp_ni = x#.cpu()
    
    x = x.masked_fill((temp_ni == 0) & (map < 0.001), 1.)

    x = x.masked_fill((temp_ni == 1) & (map < 0.001), 2.)

    x = x.masked_fill((temp_ni == 2) & (map < 0.006), 3.)
    x = x.masked_fill((temp_ni == 2) & (map >= 0.999), 1.)
    
    x = x.masked_fill((temp_ni == 3) & (map < 0.006), 4.)
    x = x.masked_fill((temp_ni == 3) & (map >= 0.975), 2.)

    x = x.masked_fill((temp_ni == 4) & (map < 0.011), 5.)
    x = x.masked_fill((temp_ni == 4) & (map >= 0.989), 3.)
    
    x = x.masked_fill((temp_ni == 5) & (map < 0.004), 6.)
    x = x.masked_fill((temp_ni == 5) & (map >= 0.999), 4.)
    
    x = x.masked_fill((temp_ni == 6) & (map < 0.028), 7.)
    x = x.masked_fill((temp_ni == 6) & (map >= 0.996), 5.)
    
    x = x.masked_fill((temp_ni == 7) & (map < 0.028), 8.)
    x = x.masked_fill((temp_ni == 7) & (map >= 0.995), 6.)

    x = x.masked_fill((temp_ni == 8) & (map < 0.051), 9.)
    x = x.masked_fill((temp_ni == 8) & (map >= 0.998), 7.)

    x = x.masked_fill((temp_ni == 9) & (map < 0.057), 10.)
    x = x.masked_fill((temp_ni == 9) & (map >= 0.998), 8.)

    x = x.masked_fill((temp_ni == 10) & (map < 0.043), 11.)
    x = x.masked_fill((temp_ni == 10) & (map >= 0.978), 9.)
    
    x = x.masked_fill((temp_ni == 11) & (map < 0.031), 12.)
    x = x.masked_fill((temp_ni == 11) & (map >= 0.998), 10.)
    
    x = x.masked_fill((temp_ni == 12) & (map < 0.003), 13.)
    x = x.masked_fill((temp_ni == 12) & (map >= 0.987), 11.)

    x = x.masked_fill((temp_ni == 13) & (map < 0.021), 14.)
    x = x.masked_fill((temp_ni == 13) & (map >= 0.997), 12.)

    x = x.masked_fill((temp_ni == 14) & (map < 0.009), 16.)
    x = x.masked_fill((temp_ni == 14) & (map >= 0.999) & (map < 0.047), 15.)
    
    x = x.masked_fill((temp_ni == 15) & (map < 0.019), 17.)
    x = x.masked_fill((temp_ni == 15) & (map >= 0.019) & (map < 0.074), 16.)
    
    x = x.masked_fill((temp_ni == 16) & (map < 0.007), 18.)
    x = x.masked_fill((temp_ni == 16) & (map >= 0.007) & (map < 0.091), 17.)
    x = x.masked_fill((temp_ni == 16) & (map >= 0.987), 15.)
    
    x = x.masked_fill((temp_ni == 17) & (map < 0.141), 18.)
    x = x.masked_fill((temp_ni == 17) & (map >= 0.963) , 16.)
    
    x = x.masked_fill((temp_ni == 18) & (map < 0.157), 19.)
    x = x.masked_fill((temp_ni == 18) & (map >= 0.93) & (map < 0.991), 17.)
    x = x.masked_fill((temp_ni == 18) & (map >= 0.991), 16.)
    
    x = x.masked_fill((temp_ni == 19) & (map < 0.004), 21.)
    x = x.masked_fill((temp_ni == 19) & (map >= 0.004) & (map < 0.161), 20.)
    x = x.masked_fill((temp_ni == 19) & (map >= 0.925), 18.)
    
    x = x.masked_fill((temp_ni == 20) & (map < 0.021), 22.)
    x = x.masked_fill((temp_ni == 20) & (map >= 0.021) & (map < 0.156), 21.)
    x = x.masked_fill((temp_ni == 20) & (map >= 0.868) & (map < 0.959), 19.)
    x = x.masked_fill((temp_ni == 20) & (map >= 0.959), 18.)

    x = x.masked_fill((temp_ni == 21) & (map < 0.012), 23.)
    x = x.masked_fill((temp_ni == 21) & (map >= 0.012) & (map < 0.151), 22.)
    x = x.masked_fill((temp_ni == 21) & (map >= 0.887) & (map < 0.992), 20.)
    x = x.masked_fill((temp_ni == 21) & (map >= 0.992), 19.)
    
    x = x.masked_fill((temp_ni == 22) & (map < 0.151), 23.)
    x = x.masked_fill((temp_ni == 22) & (map >= 0.89), 21.)
    
    x = x.masked_fill((temp_ni == 23) & (map < 0.013), 25.)
    x = x.masked_fill((temp_ni == 23) & (map >= 0.013) & (map < 0.112), 24.)
    x = x.masked_fill((temp_ni == 23) & (map >= 0.896) & (map < 0.982), 22.)
    x = x.masked_fill((temp_ni == 23) & (map >= 0.982), 21.)

    x = x.masked_fill((temp_ni == 24) & (map < 0.006), 26.)
    x = x.masked_fill((temp_ni == 24) & (map >= 0.006) & (map < 0.145), 25.)
    x = x.masked_fill((temp_ni == 24) & (map >= 0.943) & (map < 0.993), 23.)
    x = x.masked_fill((temp_ni == 24) & (map >= 0.993), 22.)

    x = x.masked_fill((temp_ni == 25) & (map < 0.001), 27.)
    x = x.masked_fill((temp_ni == 25) & (map >= 0.001) & (map < 0.144), 26.)
    x = x.masked_fill((temp_ni == 25) & (map >= 0.917), 24.)
    
    x = x.masked_fill((temp_ni == 26) & (map < 0.027), 27.)
    x = x.masked_fill((temp_ni == 26) & (map >= 0.902), 25.)
    
    x = x.masked_fill((temp_ni == 27) & (map < 0.003), 28.)
    x = x.masked_fill((temp_ni == 27) & (map >= 0.961), 26.)
    
    x = x.masked_fill((temp_ni == 28) & (map < 0.007), 29.)
    x = x.masked_fill((temp_ni == 28) & (map >= 0.987), 27.)
    
    x = x.masked_fill((temp_ni == 29) & (map < 0.01), 30.)
    x = x.masked_fill((temp_ni == 29) & (map >= 0.99), 28.)
    
    x = x.masked_fill((temp_ni == 30) & (map < 0.013), 31.)
    x = x.masked_fill((temp_ni == 30) & (map >= 0.992), 29.)
    
    x = x.masked_fill((temp_ni == 31) & (map >= 0.997), 30.)
    
    return x

def input4_qsa(x, adc_list):
    '''
    x = torch.floor( x / 5 )
    return x.clamp_max(31) * 5 + 2
    '''
    # way-2
    x = torch.floor(x / 5).clamp_max(31)
    writeout_dis('w_comb', x, adc_list)
    #output = w_nonideal(x) * 5 + 2
    output = x * 5 + 2
    
    return output

def input2_qsa(x, adc_list):
    '''
    return x.clamp_max(31)
    '''
    # way-2
    x = x.clamp_max(31)
    writeout_dis('wo_comb', x, adc_list)
    #output = wo_nonideal(x)
    output = x
    
    return output

def writeout_dis(name, convX, adc_list):
    filepath = './vgg_adc_distribution/' + name + '.dat'
    for i in list(convX.reshape(-1)):
        adc_list[int(i)] += 1
    
    if os.path.isfile(filepath):
        with open(filepath, 'r') as f:
            for i, value in enumerate(f.readlines()):
                adc_list[i] += int(value)
            f.close()
    with open(filepath, 'w') as f:
        for data in adc_list:
            f.write(str(data.item())+'\n')
        f.close()
    return 0


class ADC(torch.autograd.Function):
    @staticmethod
    def forward(ctx, convX):
        output = input2_qsa(convX)
        '''
        # non-ideal
        B, OHOW, K = convX.shape
        non_ideal_ratio = torch.tensor([0, 3, 17, 32, 64, 2980, 55, 29, 16, 4, 0], dtype=torch.float)
        non_ideal_effect = (torch.multinomial(non_ideal_ratio, B*OHOW*K, replacement=True) - 5).view(B, OHOW, K)
        output += non_ideal_effect
        '''
        return output
    
    @staticmethod
    def backward(ctx, grad_outputs):
        return grad_outputs.clone()

class ADC_comb(torch.autograd.Function):
    @staticmethod
    def forward(ctx, convX):
        output = input4_qsa(convX)
        '''
        # non-ideal
        B, OHOW, K = convX.shape
        non_ideal_ratio = torch.tensor([2, 10, 22, 33, 84, 2898, 84, 37, 17, 10, 3], dtype=torch.float)
        non_ideal_effect = (torch.multinomial(non_ideal_ratio, B*OHOW*K, replacement=True) - 5).view(B, OHOW, K)
        output += non_ideal_effect
        '''
        return output

    @staticmethod
    def backward(ctx, grad_outputs):
        return grad_outputs.clone()


def CIM_MAC(x_int, w_int, adc_list):
    B, osize, alpha = x_int.shape               # [B, OH*OW, alpha]
    alpha, K = w_int.shape                      # [alpha, K]

    # Split input bits                                # if x bit: 0b0011
    msb_x = (x_int >> 2).float()                      # msb_x: 0b00
    lsb_x = (x_int & 3).float()                       # lsb_x: 0b11
    x_float = x_int.float()
    
    # Split weight bits (upper:msb)         # if w bit: 0b01010101           
    w7 = ( (w_int>>7) & 1 ).float()                     # w7: 0b0
    w6 = ( (w_int>>6) & 1 ).float()                     # w6: 0b1
    w5 = ( (w_int>>5) & 1 ).float()                     # w5: 0b0
    w4 = ( (w_int>>4) & 1 ).float()                     # w4: 0b1
    w3 = ( (w_int>>3) & 1 ).float()                     # w3: 0b0
    w2 = ( (w_int>>2) & 1 ).float()                     # w2: 0b1
    w1 = ( (w_int>>1) & 1 ).float()                     # w1: 0b0
    w0 = ( (w_int) & 1 ).float()                        # w0: 0b1

    # Multiply and Addition
    conv7_msb = msb_x.matmul(w7)
    conv7_lsb = lsb_x.matmul(w7)
    conv6_msb = msb_x.matmul(w6)
    conv6_lsb = lsb_x.matmul(w6)
    conv5_msb = msb_x.matmul(w5)
    conv5_lsb = lsb_x.matmul(w5)
    conv4_msb = msb_x.matmul(w4)
    conv4_lsb = lsb_x.matmul(w4)
    '''
    conv3_msb = msb_x.matmul(w3)
    conv3_lsb = lsb_x.matmul(w3)
    conv2_msb = msb_x.matmul(w2)
    conv2_lsb = lsb_x.matmul(w2)
    conv1_msb = msb_x.matmul(w1)
    conv1_lsb = lsb_x.matmul(w1)
    conv0_msb = msb_x.matmul(w0)
    conv0_lsb = lsb_x.matmul(w0)
    
    conv5 = x_float.matmul(w5)
    conv4 = x_float.matmul(w4)
    '''
    conv3 = x_float.matmul(w3)
    conv2 = x_float.matmul(w2)
    conv1 = x_float.matmul(w1)
    conv0 = x_float.matmul(w0)
    
    #conv7_msb = ADC.apply(conv7_msb)
    #conv7_lsb = ADC.apply(conv7_lsb)
    #conv6_msb = ADC.apply(conv6_msb)
    #conv6_lsb = ADC.apply(conv6_lsb)
    #conv5_msb = ADC.apply(conv5_msb)
    #conv5_lsb = ADC.apply(conv5_lsb)
    #conv4_msb = ADC.apply(conv4_msb)
    #conv4_lsb = ADC.apply(conv4_lsb)
    conv7_msb = input2_qsa(conv7_msb, adc_list[0])
    conv7_lsb = input2_qsa(conv7_lsb, adc_list[0])
    conv6_msb = input2_qsa(conv6_msb, adc_list[0])
    conv6_lsb = input2_qsa(conv6_lsb, adc_list[0])
    conv5_msb = input2_qsa(conv5_msb, adc_list[0])
    conv5_lsb = input2_qsa(conv5_lsb, adc_list[0])
    conv4_msb = input2_qsa(conv4_msb, adc_list[0])
    conv4_lsb = input2_qsa(conv4_lsb, adc_list[0])
    '''
    conv3_msb = ADC.apply(conv3_msb)
    conv3_lsb = ADC.apply(conv3_lsb)
    conv2_msb = ADC.apply(conv2_msb)
    conv2_lsb = ADC.apply(conv2_lsb)
    conv1_msb = ADC.apply(conv1_msb)
    conv1_lsb = ADC.apply(conv1_lsb)
    conv0_msb = ADC.apply(conv0_msb)
    conv0_lsb = ADC.apply(conv0_lsb)
    
    conv4 = ADC_comb.apply(conv4)
    '''
    #conv3 = ADC_comb.apply(conv3)
    #conv2 = ADC_comb.apply(conv2)
    #conv1 = ADC_comb.apply(conv1)
    #conv0 = ADC_comb.apply(conv0)
    conv3 = input4_qsa(conv3, adc_list[1])
    conv2 = input4_qsa(conv2, adc_list[1])
    conv1 = input4_qsa(conv1, adc_list[1])
    conv0 = input4_qsa(conv0, adc_list[1])
    
    total = ((conv7_msb * 4 + conv7_lsb) * -128 + 
            (conv6_msb * 4 + conv6_lsb) * 64   + 
            (conv5_msb * 4 + conv5_lsb) * 32 + (conv4_msb * 4 + conv4_lsb) * 16 + (conv3) * 8  +
            (conv2) * 4  + (conv1) * 2  + (conv0))

    return total


class Conv2d_CIM_SRAM(nn.Conv2d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 bitW=32,
                 #bitA=32,
                 bitO=32,
                 adc_list=None,
                 sub_channel='v',
                 dilation=1,
                 groups=1,
                 bias=False,):
                 #padding_mode='zeros'):
        # kernel_size = _pair(kernel_size)
        # stride = _pair(stride)
        # padding = _pair(padding)
        # dilation = _pair(dilation)
        super(Conv2d_CIM_SRAM, self).__init__(in_channels, out_channels, kernel_size, stride, padding,
                                              dilation, groups, bias)
        self.bitW = bitW
        self.bitO = bitO
        self.adc_list = adc_list
        
        self.sub_flag = sub_channel == 'v' or self.in_channels==1 or self.in_channels==3
        self.fw = fw(bitW)
        self.macs = self.weight.shape[1:].numel()
        self.fo = fo(bitO)
        self.sram_cim_conv = sram_cim_conv.apply
        
    def forward(self, conv_inp, order=None):
        input = conv_inp['act']
        alpha = conv_inp['alpha']
        alpha_w = float(2 ** (self.bitW-1) - 1)
        
        q_weight = self.fw(self.weight)
 
        #conv_sub_layers = F.conv2d(input, q_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        conv_sub_layers_sram = self.sram_cim_conv(input, q_weight, self.stride[0], self.padding[0], alpha, alpha_w, self.adc_list)
        
        outputs = self.fo(conv_sub_layers_sram)
        return outputs



