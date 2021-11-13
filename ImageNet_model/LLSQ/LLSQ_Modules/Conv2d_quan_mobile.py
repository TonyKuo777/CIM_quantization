# %%
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Function, Variable
from torch.nn import Parameter
from torch.distributions import Bernoulli
from torch.nn.modules.utils import _pair
from torch.nn import init
import math
import os
import fileinput




__all__ = [
    'QuantConv2d', 'RoundFn_LLSQ', 'RoundFn_Bias', 'quan_alpha'
]


class RoundFn_LLSQ(Function):
    @staticmethod
    def forward(ctx, input, alpha, pwr_coef, bit):
        # the standard quantization function quantized to k bit, where 2^k=pwr_coef, the input must scale to [0,1]
        
        # alpha = quan_alpha(alpha, 16)
        x_alpha_div = (input  / alpha  ).round().clamp( min =-(pwr_coef), max = (pwr_coef-1)) * alpha
        
        ctx.pwr_coef = pwr_coef
        ctx.bit      = bit
        ctx.save_for_backward(input, alpha)
        return x_alpha_div 
    @staticmethod
    def backward(ctx, grad_output):
        input, alpha = ctx.saved_tensors
        pwr_coef = ctx.pwr_coef
        bit      = ctx.bit
        #alpha = quan_alpha(alpha, 16)
        quan_Em =  (input  / (alpha ) ).round().clamp( min =-(pwr_coef), max = (pwr_coef-1)) * alpha  
        quan_El =  (input / ((alpha ) / 2) ).round().clamp( min =-(pwr_coef), max = (pwr_coef-1)) * (alpha  / 2) 
        quan_Er = (input / ((alpha ) * 2) ).round().clamp( min =-(pwr_coef), max = (pwr_coef-1)) * (alpha  * 2) 
        
        if list(alpha.size())[0] > 1:
            El = torch.sum(torch.pow((input - quan_El), 2 ), dim = 0)
            Er = torch.sum(torch.pow((input - quan_Er), 2 ), dim = 0)
            Em = torch.sum(torch.pow((input - quan_Em), 2 ), dim = 0)
            
            d_better = torch.argmin( torch.stack([El, Em, Er], dim=0), dim=0) - 1
            delta_G = - (torch.pow(alpha , 2)) * ( d_better)
        else:
            El = torch.sum(torch.pow((input - quan_El), 2 ))
            Er = torch.sum(torch.pow((input - quan_Er), 2 ))
            Em = torch.sum(torch.pow((input - quan_Em), 2 ))
            d_better = torch.Tensor([El, Em, Er]).argmin() -1
            delta_G = (-1) * (torch.pow(alpha , 2)) * ( d_better) 
        
        grad_input = grad_output.clone()
        return  grad_input, delta_G, None, None


class RoundFn_Bias(Function):
    @staticmethod
    def forward(ctx, input, alpha, pwr_coef, bit):
        ctx.save_for_backward(input, alpha)
        # alpha = quan_alpha(alpha, 16)
        alpha = torch.reshape(alpha, (-1,))
        # alpha = quan_alpha(alpha, bit)
        x_alpha_div = (input  / alpha).round().clamp( min =-(pwr_coef), max = (pwr_coef-1)) * alpha 
        ctx.pwr_coef = pwr_coef
        
        return x_alpha_div 
    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        
        return  grad_input, None, None, None


class QuantConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=False, bit=32, extern_init=False, init_model=nn.Sequential()):
        super(QuantConv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            groups, bias)
        self.bit = bit
        self.pwr_coef =  2**(bit - 1) 
        self.Round_w = RoundFn_LLSQ.apply
        self.Round_b = RoundFn_Bias.apply
        self.bias_flag = bias
        #self.alpha_w = Variable(torch.rand( out_channels,1,1,1)).cuda()
        #self.alpha_w = Parameter(torch.rand( out_channels))
        #self.alpha_qfn = quan_fn_alpha()
        nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')
        if extern_init:
            param=list(init_model.parameters())
            self.weight=Parameter(param[0])
            if bias:
                self.bias=Parameter(param[1])
        if bit < 0:
            self.alpha_w = None
            self.init_state = 0
        else:
            self.alpha_w = Parameter(torch.rand( out_channels))
            self.register_buffer('init_state', torch.zeros(1))
        # self.init_state = 0
    def forward(self, x):
        if self.bit == 32:
            return F.conv2d(
                x, self.weight, self.bias, self.stride, self.padding,
                self.dilation, self.groups)
        else:
            w_reshape = self.weight.reshape([self.weight.shape[0], -1]).transpose(0, 1)
            if self.training and self.init_state == 0:            
                self.alpha_w.data.copy_(w_reshape.detach().abs().max(dim=0)[0] / self.pwr_coef)
                self.init_state.fill_(1)
               
            #assert not torch.isnan(x).any(), "Conv2d Input should not be 'nan'"
            #self.alpha_qfn(self.alpha_w)
            #if torch.isnan(self.alpha_w).any() or torch.isinf(self.alpha_w).any():
            #    assert not torch.isnan(wq).any(), self.alpha_w
            #    assert not torch.isinf(wq).any(), self.alpha_w

            wq =  self.Round_w(w_reshape, self.alpha_w, self.pwr_coef, self.bit)
            w_q = wq.transpose(0, 1).reshape(self.weight.shape)

            if self.bias_flag == True:
                LLSQ_b  = self.Round_b(self.bias, self.alpha_w, self.pwr_coef, self.bit)
            else:
                LLSQ_b = self.bias
            
            # assert not torch.isnan(self.weight).any(), "Weight should not be 'nan'"
            # if torch.isnan(wq).any() or torch.isinf(wq).any():
            #     print(self.alpha_w)
            #     assert not torch.isnan(wq).any(), "Conv2d Weights should not be 'nan'"
            #     assert not torch.isinf(wq).any(), "Conv2d Weights should not be 'nan'"
            
            return F.conv2d(
                x,  w_q, LLSQ_b, self.stride, self.padding, self.dilation,
                self.groups)
    def extra_repr(self):
        s_prefix = super(QuantConv2d, self).extra_repr()
        if self.alpha_w is None:
            return '{}, fake'.format(s_prefix)
        return '{}'.format(s_prefix)

def quan_alpha(alpha, bits):
    if(bits==32):
        alpha_q = alpha
    else:
        q_code  = bits - torch.ceil( torch.log2( torch.max(alpha)) + 1 - 1e-5 )
        alpha_q = torch.clamp( torch.round( alpha * (2**q_code)), -2**(bits - 1), 2**(bits - 1) - 1 ) / (2**q_code)
    return alpha_q

class quan_fn_alpha(nn.Module):
    def __init__(self,  bit=32 ):
        super(quan_fn_alpha, self).__init__()
        self.bits = bit
        self.pwr_coef   = 2 ** (bit - 1)
    def forward(self, alpha):
        q_code  = self.bits - torch.ceil( torch.log2( torch.max(alpha)) + 1 - 1e-5 )
        alpha_q = torch.clamp( torch.round( alpha * (2**q_code)), -2**(self.bits - 1), 2**(self.bits - 1) - 1 ) / (2**q_code)
        return alpha_q
    def backward(self, input):
        return input

class sram_cim_conv(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, stride, padding, scale_alpha, scale_alpha_w, bias=None, adc_list=None):
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
        ctx.save_for_backward(input, weight, bias)
        ctx.stride = stride
        ctx.padding = padding 

        B, CH, H, W = input.shape
        K, CH, KH, KW = weight.shape
        alpha = 16                                                  # alpha is the MAC number in CIM
        
        if (CH % alpha) == 0:
            # 16 channel as a Group
            GPS = CH//alpha                                             # GPS means GrouPS
            input_slices = input.view(B, GPS, alpha, H, W)              # [B, CH//alpha, 16, H, W]
            weight_slices = weight.view(K, GPS, alpha, KH, KW)          # [K, CH//alpha, 16, KH, KW]

            # Initialize the OFM
            # calculate output height and width
            OH = int( (H - KH + 2 * padding[0]) / stride[0] + 1 )
            OW = int( (W - KW + 2 * padding[1]) / stride[1] + 1 )
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
                    x_int = torch.round(input_unfold[:, :, i, :] / scale_alpha).int()   # [B, OH*OW, alpha]    255 <- 2**8 - 1
                    w_int = torch.round(torch.clamp(weight_unfold[i, :, :] / scale_alpha_w, min=-128, max=127)).int()   # [-128, 127]
                    
                    msb_x_int = x_int >> 4
                    lsb_x_int = x_int & 15

                    output_unfold += ( (CIM_MAC(msb_x_int, w_int, adc_list) << 4) + CIM_MAC(lsb_x_int, w_int, adc_list) )

                output_unfold = (output_unfold * scale_alpha_w).transpose(1, 2)                            # [B, K, OH*OW]
                output += torch.nn.functional.fold(output_unfold, (OH, OW), (1, 1))
            output = output * scale_alpha
            return output
        else:
            # 16 channel as a Group
            input_slices = torch.split(input, alpha, dim=1)             # ([B, 16, H, W], [B, 8, H, W])
            weight_slices = torch.split(weight, alpha, dim=1)           # ([K, 16, KH, KW], [K, 8, KH, KW])
            
            GPS = len(input_slices)

            # Initialize the OFM
            # calculate output height and width
            OH = int( (H - KH + 2 * padding[0]) / stride[0] + 1 )
            OW = int( (W - KW + 2 * padding[1]) / stride[1] + 1 )
            output = torch.zeros((B, K, OH, OW))                                        # [B, K, OH, OW]

            for gp in range(GPS):
                if gp == GPS-1:
                    alpha = 8
                input_unfold = torch.nn.functional.unfold(input_slices[gp],             # [B, alpha*KH*KW, OH*OW]
                                                        kernel_size=(KH, KW), 
                                                        stride=stride, 
                                                        padding=padding)
                input_unfold = input_unfold.transpose(1, 2)                             # [B, OH*OW, alpha*KH*KW]
                input_unfold = input_unfold.view(B, OH*OW, KH*KW, alpha)                # [B, OH*OW, KH*KW, alpha]

                weight_unfold = weight_slices[gp].view(K, -1).t()                       # [alpha*KH*KW, K]
                weight_unfold = weight_unfold.view(KH*KW, alpha, K)                     # [KH*KW, alpha, K]

                output_unfold = torch.zeros((B, OH*OW, K))                              # [B, OH*OW, K]

                for i in range(KH*KW):
                    # 8a8w
                    # FP --> Int
                    x_int = torch.round(input_unfold[:, :, i, :] / scale_alpha).int()   # [B, OH*OW, alpha]    255 <- 2**8 - 1
                    w_int = torch.round(torch.clamp(weight_unfold[i, :, :] / scale_alpha_w, min=-128, max=127)).int()   # [-128, 127]
                    
                    msb_x_int = x_int >> 4
                    lsb_x_int = x_int & 15

                    output_unfold += ( (CIM_MAC(msb_x_int, w_int, adc_list) << 4) + CIM_MAC(lsb_x_int, w_int, adc_list) )

                output_unfold = (output_unfold * scale_alpha_w).transpose(1, 2)                            # [B, K, OH*OW]
                output += torch.nn.functional.fold(output_unfold, (OH, OW), (1, 1))
            output = output * scale_alpha
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
            return grad_input, grad_weight, None, None, None, grad_bias
    
        return grad_input, grad_weight, None, None, None, None, None, None

class sram_cim_conv_dw(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, stride, padding, scale_alpha, scale_alpha_w, bias=None, adc_list=None):
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
        ctx.save_for_backward(input, weight, bias)
        ctx.stride = stride
        ctx.padding = padding
        
        B, CH, H, W = input.shape
        CH, _, KH, KW = weight.shape

        output_dw = torch.zeros((B, 1, H, W))
        counter = 0
        
        for ch in range(CH):
            input_slices = input[:, ch, :, :].view(B, 1, H, W)              # [B, 1, H, W]
            weight_slices = weight[ch, :, :].view(KH, KW)                # [KH, KW]

            # Initialize the OFM
            # calculate output height and width
            OH = int( (H - KH + 2 * padding[0]) / stride[0] + 1 )
            OW = int( (W - KW + 2 * padding[1]) / stride[1] + 1 )
            
            input_unfold = torch.nn.functional.unfold(input_slices,                 # [B, 1*KH*KW, OH*OW]
                                                    kernel_size=(KH, KW), 
                                                    stride=stride, 
                                                    padding=padding)
            input_unfold = input_unfold.transpose(1, 2)                      # [B, OH*OW, 1*KH*KW]
            input_unfold = input_unfold.view(B, OH*OW, KH*KW)                # [B, OH*OW, KH*KW]
            
            weight_unfold = weight_slices.view(1, -1).t()                       # [KH*KW]
            weight_unfold = weight_unfold.view(KH*KW, 1)                     # [KH*KW, 1]
                
            # 8a8w
            # FP --> Int
            x_int = torch.round(input_unfold / scale_alpha).int()                                       # [B, OH*OW, KH*KW, 1]      255 <- 2**8 - 1
            w_int = torch.round(torch.clamp(weight_unfold / scale_alpha_w[ch], min=-128, max=127)).int()    # [KH*KW, 1]             128 <- 2**(8-1)
            
            msb_x_int = x_int >> 4
            lsb_x_int = x_int & 15
            
            output_unfold = (CIM_MAC_dw(msb_x_int, w_int, adc_list) << 4) + CIM_MAC_dw(lsb_x_int, w_int, adc_list)         # [B, OH*OW, 1]
            
            output_unfold = (output_unfold * scale_alpha_w[ch] * scale_alpha).transpose(1, 2)                            # [B, 1, OH*OW]
            output = torch.nn.functional.fold(output_unfold, (OH, OW), (1, 1))                         # [B, 1, OH, OW]
            
            if counter == 0:
                output_dw = output
                counter = 1
            else:
                output_dw = torch.cat((output_dw, output), dim=1)

        return output_dw
  
    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_variables
        stride = ctx.stride
        padding = ctx.padding
        
        grad_input = grad_weight = grad_bias = None

        if ctx.needs_input_grad[0]:
            grad_input = torch.nn.grad.conv2d_input(input.shape, weight, grad_output, stride, padding, groups=input.shape[1])
        
        if ctx.needs_input_grad[1]:
            grad_weight = torch.nn.grad.conv2d_weight(input, weight.shape, grad_output, stride, padding)
            
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = None
        if bias is not None:
            return grad_input, grad_weight, None, None, grad_bias
    
        return grad_input, grad_weight, None, None, None, None, None, None

def w_nonideal(x):
    map = torch.rand(x.shape)
    temp = x  #.cpu()
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
    x = x.masked_fill((temp == 2) & (map >= 0.971) & (map < 0.991), 3.)
    x = x.masked_fill((temp == 2) & (map >= 0.991), 4.)
    
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
    # (ImNet-Mobile) Revise non ideal model #
    ########################################
    #map = torch.rand(x.shape)
    temp_ni = x#.cpu()
    
    x = x.masked_fill((temp_ni == 0) & (map < 0.006), 1.)

    x = x.masked_fill((temp_ni == 1) & (map < 0.001), 2.)

    x = x.masked_fill((temp_ni == 2) & (map < 0.004), 3.)
    
    x = x.masked_fill((temp_ni == 3) & (map < 0.024), 5.)
    x = x.masked_fill((temp_ni == 3) & (map >= 0.024) & (map < 0.05), 4.)
    x = x.masked_fill((temp_ni == 3) & (map >= 0.995), 2.)
    
    x = x.masked_fill((temp_ni == 4) & (map < 0.054), 5.)
    x = x.masked_fill((temp_ni == 4) & (map >= 0.885) & (map < 0.942), 3.)
    x = x.masked_fill((temp_ni == 4) & (map >= 0.942), 2.)

    x = x.masked_fill((temp_ni == 5) & (map < 0.008), 7.)
    x = x.masked_fill((temp_ni == 5) & (map >= 0.008) & (map < 0.027), 6.)
    x = x.masked_fill((temp_ni == 5) & (map >= 0.978) & (map < 0.99), 4.)
    x = x.masked_fill((temp_ni == 5) & (map >= 0.99), 3.)

    x = x.masked_fill((temp_ni == 6) & (map < 0.01), 8.)
    x = x.masked_fill((temp_ni == 6) & (map >= 0.01) & (map < 0.028), 7.)
    x = x.masked_fill((temp_ni == 6) & (map >= 0.985) & (map < 0.998), 5.)
    x = x.masked_fill((temp_ni == 6) & (map >= 0.998), 4.)

    x = x.masked_fill((temp_ni == 7) & (map < 0.005), 9.)
    x = x.masked_fill((temp_ni == 7) & (map >= 0.005) & (map < 0.023), 8.)
    x = x.masked_fill((temp_ni == 7) & (map >= 0.985) & (map < 0.996), 6.)
    x = x.masked_fill((temp_ni == 7) & (map >= 0.996), 5.)

    x = x.masked_fill((temp_ni == 8) & (map < 0.004), 10.)
    x = x.masked_fill((temp_ni == 8) & (map >= 0.004) & (map < 0.058), 9.)
    x = x.masked_fill((temp_ni == 8) & (map >= 0.981) & (map < 0.999), 7.)
    x = x.masked_fill((temp_ni == 8) & (map >= 0.999), 6.)

    x = x.masked_fill((temp_ni == 9) & (map < 0.007), 11.)
    x = x.masked_fill((temp_ni == 9) & (map >= 0.007) & (map < 0.056), 10.)
    x = x.masked_fill((temp_ni == 9) & (map >= 0.987) & (map < 0.996), 8.)
    x = x.masked_fill((temp_ni == 9) & (map >= 0.996), 7.)

    x = x.masked_fill((temp_ni == 10) & (map < 0.001), 12.)
    x = x.masked_fill((temp_ni == 10) & (map >= 0.001) & (map < 0.033), 11.)
    x = x.masked_fill((temp_ni == 10) & (map >= 0.978), 9.)

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

    x = x.masked_fill((temp_ni == 14) & (map < 0.011), 16.)
    x = x.masked_fill((temp_ni == 14) & (map >= 0.011) & (map < 0.05), 15.)
    x = x.masked_fill((temp_ni == 14) & (map >= 0.99) & (map < 0.999), 13.)
    x = x.masked_fill((temp_ni == 14) & (map >= 0.999), 12.)

    x = x.masked_fill((temp_ni == 15) & (map < 0.015), 17.)
    x = x.masked_fill((temp_ni == 15) & (map >= 0.015) & (map < 0.073), 16.)
    
    x = x.masked_fill((temp_ni == 16) & (map < 0.014), 18.)
    x = x.masked_fill((temp_ni == 16) & (map >= 0.014) & (map < 0.091), 17.)
    x = x.masked_fill((temp_ni == 16) & (map >= 0.993), 15.)

    x = x.masked_fill((temp_ni == 17) & (map < 0.002), 20.)
    x = x.masked_fill((temp_ni == 17) & (map >= 0.002) & (map < 0.01), 19.)
    x = x.masked_fill((temp_ni == 17) & (map >= 0.01) & (map < 0.13), 18.)
    x = x.masked_fill((temp_ni == 17) & (map >= 0.943), 16.)

    x = x.masked_fill((temp_ni == 18) & (map < 0.012), 21.)
    x = x.masked_fill((temp_ni == 18) & (map >= 0.012) & (map < 0.036), 20.)
    x = x.masked_fill((temp_ni == 18) & (map >= 0.036) & (map < 0.157), 19.)
    x = x.masked_fill((temp_ni == 18) & (map >= 0.924) & (map < 0.976), 17.)
    x = x.masked_fill((temp_ni == 18) & (map >= 0.976), 16.)

    x = x.masked_fill((temp_ni == 19) & (map < 0.003), 22.)
    x = x.masked_fill((temp_ni == 19) & (map >= 0.003) & (map < 0.067), 21.)
    x = x.masked_fill((temp_ni == 19) & (map >= 0.067) & (map < 0.195), 20.)
    x = x.masked_fill((temp_ni == 19) & (map >= 0.848) & (map < 0.95), 18.)
    x = x.masked_fill((temp_ni == 19) & (map >= 0.95), 17.)

    x = x.masked_fill((temp_ni == 20) & (map < 0.042), 23.)
    x = x.masked_fill((temp_ni == 20) & (map >= 0.042) & (map < 0.098), 22.)
    x = x.masked_fill((temp_ni == 20) & (map >= 0.098) & (map < 0.211), 21.)
    x = x.masked_fill((temp_ni == 20) & (map >= 0.817) & (map < 0.946), 19.)
    x = x.masked_fill((temp_ni == 20) & (map >= 0.946), 18.)

    x = x.masked_fill((temp_ni == 21) & (map < 0.062), 24.)
    x = x.masked_fill((temp_ni == 21) & (map >= 0.062) & (map < 0.133), 23.)
    x = x.masked_fill((temp_ni == 21) & (map >= 0.133) & (map < 0.252), 22.)
    x = x.masked_fill((temp_ni == 21) & (map >= 0.778) & (map < 0.923), 20.)
    x = x.masked_fill((temp_ni == 21) & (map >= 0.923), 19.)

    x = x.masked_fill((temp_ni == 22) & (map < 0.069), 25.)
    x = x.masked_fill((temp_ni == 22) & (map >= 0.069) & (map < 0.145), 24.)
    x = x.masked_fill((temp_ni == 22) & (map >= 0.145) & (map < 0.282), 23.)
    x = x.masked_fill((temp_ni == 22) & (map >= 0.709) & (map < 0.854), 21.)
    x = x.masked_fill((temp_ni == 22) & (map >= 0.854) & (map < 0.927), 20.)
    x = x.masked_fill((temp_ni == 22) & (map >= 0.927), 19.)

    x = x.masked_fill((temp_ni == 23) & (map < 0.076), 26.)
    x = x.masked_fill((temp_ni == 23) & (map >= 0.076) & (map < 0.153), 25.)
    x = x.masked_fill((temp_ni == 23) & (map >= 0.153) & (map < 0.255), 24.)
    x = x.masked_fill((temp_ni == 23) & (map >= 0.726) & (map < 0.84), 22.)
    x = x.masked_fill((temp_ni == 23) & (map >= 0.84) & (map < 0.923), 21.)
    x = x.masked_fill((temp_ni == 23) & (map >= 0.923), 20.)

    x = x.masked_fill((temp_ni == 24) & (map < 0.114), 26.)
    x = x.masked_fill((temp_ni == 24) & (map >= 0.114) & (map < 0.24), 25.)
    x = x.masked_fill((temp_ni == 24) & (map >= 0.618) & (map < 0.762), 23.)
    x = x.masked_fill((temp_ni == 24) & (map >= 0.762) & (map < 0.885), 22.)
    x = x.masked_fill((temp_ni == 24) & (map >= 0.885), 21.)

    x = x.masked_fill((temp_ni == 25) & (map < 0.155), 27.)
    x = x.masked_fill((temp_ni == 25) & (map >= 0.155) & (map < 0.305), 26.)
    x = x.masked_fill((temp_ni == 25) & (map >= 0.52) & (map < 0.694), 24.)
    x = x.masked_fill((temp_ni == 25) & (map >= 0.694) & (map < 0.848), 23.)
    x = x.masked_fill((temp_ni == 25) & (map >= 0.848), 22.)
    
    # Actually, "DIV/0!"
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
    temp = x  #.cpu()
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
    x = x.masked_fill((temp == 2) & (map >= 0.981), 3.)
    
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
    # (ImNet-Mobile) Revise non ideal model #
    ########################################
    
    #map = torch.rand(x.shape)
    temp_ni = x#.cpu()
    
    x = x.masked_fill((temp_ni == 0) & (map < 0.002), 1.)

    x = x.masked_fill((temp_ni == 1) & (map < 0.001), 2.)

    x = x.masked_fill((temp_ni == 2) & (map < 0.007), 3.)
    x = x.masked_fill((temp_ni == 2) & (map >= 0.999), 1.)
    
    x = x.masked_fill((temp_ni == 3) & (map < 0.015), 4.)
    x = x.masked_fill((temp_ni == 3) & (map >= 0.968), 2.)

    x = x.masked_fill((temp_ni == 4) & (map < 0.014), 5.)
    x = x.masked_fill((temp_ni == 4) & (map >= 0.987), 3.)
    
    x = x.masked_fill((temp_ni == 5) & (map < 0.004), 6.)
    x = x.masked_fill((temp_ni == 5) & (map >= 0.999), 4.)
    
    x = x.masked_fill((temp_ni == 6) & (map < 0.029), 7.)
    x = x.masked_fill((temp_ni == 6) & (map >= 0.998), 5.)
    
    x = x.masked_fill((temp_ni == 7) & (map < 0.03), 8.)
    x = x.masked_fill((temp_ni == 7) & (map >= 0.994), 6.)

    x = x.masked_fill((temp_ni == 8) & (map < 0.052), 9.)
    
    x = x.masked_fill((temp_ni == 9) & (map < 0.055), 10.)

    x = x.masked_fill((temp_ni == 10) & (map < 0.046), 11.)
    x = x.masked_fill((temp_ni == 10) & (map >= 0.979), 9.)
    
    x = x.masked_fill((temp_ni == 11) & (map < 0.026), 12.)
    
    x = x.masked_fill((temp_ni == 12) & (map < 0.006), 13.)
    x = x.masked_fill((temp_ni == 12) & (map >= 0.983), 11.)

    x = x.masked_fill((temp_ni == 13) & (map < 0.021), 14.)
    x = x.masked_fill((temp_ni == 13) & (map >= 0.997), 12.)

    x = x.masked_fill((temp_ni == 14) & (map < 0.004), 16.)
    x = x.masked_fill((temp_ni == 14) & (map >= 0.004) & (map < 0.038), 15.)
    
    x = x.masked_fill((temp_ni == 15) & (map < 0.009), 17.)
    x = x.masked_fill((temp_ni == 15) & (map >= 0.009) & (map < 0.055), 16.)
    
    x = x.masked_fill((temp_ni == 16) & (map < 0.001), 18.)
    x = x.masked_fill((temp_ni == 16) & (map >= 0.001) & (map < 0.081), 17.)
    x = x.masked_fill((temp_ni == 16) & (map >= 0.992), 15.)
    
    x = x.masked_fill((temp_ni == 17) & (map < 0.004), 19.)
    x = x.masked_fill((temp_ni == 17) & (map >= 0.004) & (map < 0.13), 18.)
    x = x.masked_fill((temp_ni == 17) & (map >= 0.959) , 16.)
    
    x = x.masked_fill((temp_ni == 18) & (map < 0.019), 20.)
    x = x.masked_fill((temp_ni == 18) & (map >= 0.019) & (map < 0.186), 19.)
    x = x.masked_fill((temp_ni == 18) & (map >= 0.886) & (map < 0.968), 17.)
    x = x.masked_fill((temp_ni == 18) & (map >= 0.968), 16.)
    
    x = x.masked_fill((temp_ni == 19) & (map < 0.029), 21.)
    x = x.masked_fill((temp_ni == 19) & (map >= 0.029) & (map < 0.175), 20.)
    x = x.masked_fill((temp_ni == 19) & (map >= 0.912), 18.)
    
    x = x.masked_fill((temp_ni == 20) & (map < 0.056), 22.)
    x = x.masked_fill((temp_ni == 20) & (map >= 0.056) & (map < 0.202), 21.)
    x = x.masked_fill((temp_ni == 20) & (map >= 0.797) & (map < 0.947), 19.)
    x = x.masked_fill((temp_ni == 20) & (map >= 0.947), 18.)

    x = x.masked_fill((temp_ni == 21) & (map < 0.052), 23.)
    x = x.masked_fill((temp_ni == 21) & (map >= 0.052) & (map < 0.206), 22.)
    x = x.masked_fill((temp_ni == 21) & (map >= 0.824) & (map < 0.949), 20.)
    x = x.masked_fill((temp_ni == 21) & (map >= 0.949), 19.)
    
    x = x.masked_fill((temp_ni == 22) & (map < 0.185), 23.)
    x = x.masked_fill((temp_ni == 22) & (map >= 0.839), 21.)
    
    x = x.masked_fill((temp_ni == 23) & (map < 0.063), 25.)
    x = x.masked_fill((temp_ni == 23) & (map >= 0.063) & (map < 0.184), 24.)
    x = x.masked_fill((temp_ni == 23) & (map >= 0.809) & (map < 0.931), 22.)
    x = x.masked_fill((temp_ni == 23) & (map >= 0.931), 21.)

    x = x.masked_fill((temp_ni == 24) & (map < 0.071), 26.)
    x = x.masked_fill((temp_ni == 24) & (map >= 0.071) & (map < 0.239), 25.)
    x = x.masked_fill((temp_ni == 24) & (map >= 0.826) & (map < 0.928), 23.)
    x = x.masked_fill((temp_ni == 24) & (map >= 0.928), 22.)

    x = x.masked_fill((temp_ni == 25) & (map < 0.062), 27.)
    x = x.masked_fill((temp_ni == 25) & (map >= 0.062) & (map < 0.244), 26.)
    x = x.masked_fill((temp_ni == 25) & (map >= 0.879), 24.)
    
    x = x.masked_fill((temp_ni == 26) & (map < 0.073), 27.)
    x = x.masked_fill((temp_ni == 26) & (map >= 0.875), 25.)
    
    x = x.masked_fill((temp_ni == 27) & (map < 0.072), 28.)
    x = x.masked_fill((temp_ni == 27) & (map >= 0.896), 26.)
    
    x = x.masked_fill((temp_ni == 28) & (map < 0.014), 29.)
    x = x.masked_fill((temp_ni == 28) & (map >= 0.981), 27.)
    
    x = x.masked_fill((temp_ni == 29) & (map < 0.057), 30.)
    x = x.masked_fill((temp_ni == 29) & (map >= 0.932), 28.)
    
    # Actually, "DIV/0!"
    x = x.masked_fill((temp_ni == 30) & (map < 0.013), 31.)
    x = x.masked_fill((temp_ni == 30) & (map >= 0.992), 29.)
    
    x = x.masked_fill((temp_ni == 31) & (map >= 0.997), 30.)
    
    return x

def input4_qsa(x):
    '''
    x = torch.floor( x / 5 )
    return x.clamp_max(31) * 5 + 2
    '''
    # way-2
    x = torch.floor(x / 5).clamp_max(31)
    #output = w_nonideal(x) * 5 + 2
    #writeout_dis('w_comb', x, adc_list)
    
    return output
    
def input2_qsa(x):
    '''
    return x.clamp_max(31)
    '''
    # way-2
    x = x.clamp_max(31)
    #writeout_dis('wo_comb', output, adc_list)
    output = wo_nonideal(x)
    
    return output
    
class ADC(torch.autograd.Function):
    @staticmethod
    def forward(ctx, convX):
        output = input2_qsa(convX)
        return output
    
    @staticmethod
    def backward(ctx, grad_outputs):
        return grad_outputs.clone()

class ADC_comb(torch.autograd.Function):
    @staticmethod
    def forward(ctx, convX):
        output = input4_qsa(convX)
        return output

    @staticmethod
    def backward(ctx, grad_outputs):
        return grad_outputs.clone()
    
def input4_qsa_dw(x):
    '''
    x = torch.floor( x / 5 )
    return x.clamp_max(31) * 5 + 2
    '''
    # way-2
    x = torch.floor(x / 5).clamp_max(31)
    #output = w_nonideal(x) * 5 + 2
    output = x * 5 + 2
    
    return output
    
def input2_qsa_dw(x):
    '''
    return x.clamp_max(31)
    '''
    # way-2
    output = x.clamp_max(31)
    #output = wo_nonideal(x)
    #output = x
    
    return output

def writeout_dis(name, convX, adc_list):
    filepath = './mobile_adc_distribution/' + name + '.dat'
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

class ADC_dw(torch.autograd.Function):
    @staticmethod
    def forward(ctx, convX):
        output = input2_qsa_dw(convX)
        return output
    
    @staticmethod
    def backward(ctx, grad_outputs):
        return grad_outputs.clone()

class ADC_comb_dw(torch.autograd.Function):
    @staticmethod
    def forward(ctx, convX):
        output = input4_qsa_dw(convX)
        return output

    @staticmethod
    def backward(ctx, grad_outputs):
        return grad_outputs.clone()
    
def CIM_MAC(x_int, w_int, adc_list=None):
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
    conv3_msb = msb_x.matmul(w3)
    conv3_lsb = lsb_x.matmul(w3)
    conv2_msb = msb_x.matmul(w2)
    conv2_lsb = lsb_x.matmul(w2)
    '''
    conv1_msb = msb_x.matmul(w1)
    conv1_lsb = lsb_x.matmul(w1)
    conv0_msb = msb_x.matmul(w0)
    conv0_lsb = lsb_x.matmul(w0)
    
    conv5 = x_float.matmul(w5)
    conv4 = x_float.matmul(w4)
    conv3 = x_float.matmul(w3)
    conv2 = x_float.matmul(w2)
    '''
    conv1 = x_float.matmul(w1)
    conv0 = x_float.matmul(w0)
    
    conv7_msb = input2_qsa(conv7_msb)
    conv7_lsb = input2_qsa(conv7_lsb)
    conv6_msb = input2_qsa(conv6_msb)
    conv6_lsb = input2_qsa(conv6_lsb)
    conv5_msb = input2_qsa(conv5_msb)
    conv5_lsb = input2_qsa(conv5_lsb)
    conv4_msb = input2_qsa(conv4_msb)
    conv4_lsb = input2_qsa(conv4_lsb)
    conv3_msb = input2_qsa(conv3_msb)
    conv3_lsb = input2_qsa(conv3_lsb)
    conv2_msb = input2_qsa(conv2_msb)
    conv2_lsb = input2_qsa(conv2_lsb)
    '''
    conv1_msb = ADC.apply(conv1_msb)
    conv1_lsb = ADC.apply(conv1_lsb)
    conv0_msb = ADC.apply(conv0_msb)
    conv0_lsb = ADC.apply(conv0_lsb)
    
    conv5 = ADC_comb.apply(conv5)
    conv4 = ADC_comb.apply(conv4)
    conv3 = ADC_comb.apply(conv3)
    conv2 = ADC_comb.apply(conv2)
    '''
    conv1 = input4_qsa(conv1)
    conv0 = input4_qsa(conv0)
    
    total = ((conv7_msb * 4 + conv7_lsb) * -128 + 
            (conv6_msb * 4 + conv6_lsb) * 64   + 
            (conv5_msb * 4 + conv5_lsb) * 32 + (conv4_msb * 4 + conv4_lsb) * 16 + (conv3_msb * 4 + conv3_lsb) * 8  +
            (conv2_msb * 4 + conv2_lsb) * 4  + (conv1) * 2  + (conv0))

    return total

def CIM_MAC_dw(x_int, w_int, adc_list):
    B, osize, KHKW = x_int.shape               # [B, OH*OW, KH*KW]
    KHKW, _ = w_int.shape                      # [KH*KW, 1]

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

    # Multiply and Addition                 # [B, OH*OW, 1]
    conv7_msb = msb_x.matmul(w7)
    conv7_lsb = lsb_x.matmul(w7)
    conv6_msb = msb_x.matmul(w6)
    conv6_lsb = lsb_x.matmul(w6)
    conv5_msb = msb_x.matmul(w5)
    conv5_lsb = lsb_x.matmul(w5)
    conv4_msb = msb_x.matmul(w4)
    conv4_lsb = lsb_x.matmul(w4)
    conv3_msb = msb_x.matmul(w3)
    conv3_lsb = lsb_x.matmul(w3)
    conv2_msb = msb_x.matmul(w2)
    conv2_lsb = lsb_x.matmul(w2)
    '''
    conv1_msb = msb_x.matmul(w1)
    conv1_lsb = lsb_x.matmul(w1)
    conv0_msb = msb_x.matmul(w0)
    conv0_lsb = lsb_x.matmul(w0)
    
    conv5 = x_float.matmul(w5)
    conv4 = x_float.matmul(w4)
    conv3 = x_float.matmul(w3)
    conv2 = x_float.matmul(w2)
    '''
    conv1 = x_float.matmul(w1)
    conv0 = x_float.matmul(w0)
    
    # Variation
    conv7_msb = input2_qsa_dw(conv7_msb)
    conv7_lsb = input2_qsa_dw(conv7_lsb)
    conv6_msb = input2_qsa_dw(conv6_msb)
    conv6_lsb = input2_qsa_dw(conv6_lsb)
    conv5_msb = input2_qsa_dw(conv5_msb)
    conv5_lsb = input2_qsa_dw(conv5_lsb)
    conv4_msb = input2_qsa_dw(conv4_msb)
    conv4_lsb = input2_qsa_dw(conv4_lsb)
    conv3_msb = input2_qsa_dw(conv3_msb)
    conv3_lsb = input2_qsa_dw(conv3_lsb)
    conv2_msb = input2_qsa_dw(conv2_msb)
    conv2_lsb = input2_qsa_dw(conv2_lsb)
    '''
    conv1_msb = ADC.apply(conv1_msb)
    conv1_lsb = ADC.apply(conv1_lsb)
    conv0_msb = ADC.apply(conv0_msb)
    conv0_lsb = ADC.apply(conv0_lsb)
    
    conv5 = ADC_comb.apply(conv5)
    conv4 = ADC_comb.apply(conv4)
    conv3 = ADC_comb.apply(conv3)
    conv2 = ADC_comb.apply(conv2)
    '''
    conv1 = input4_qsa_dw(conv1)
    conv0 = input4_qsa_dw(conv0)
    
    total = ((conv7_msb * 4 + conv7_lsb) * -128 + 
            (conv6_msb * 4 + conv6_lsb) * 64   + 
            (conv5_msb * 4 + conv5_lsb) * 32 + (conv4_msb * 4 + conv4_lsb) * 16 + (conv3_msb * 4 + conv3_lsb) * 8  +
            (conv2_msb * 4 + conv2_lsb) * 4  + (conv1) * 2  + (conv0))

    return total

class Conv2d_CIM_SRAM(nn.Conv2d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size, 
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1, 
                 bias=False,
                 bit=32,
                 extern_init=False,
                 init_model=nn.Sequential(),
                adc_list=None):
        super(Conv2d_CIM_SRAM, self).__init__(in_channels, out_channels, kernel_size, stride, padding,
                                              dilation, groups, bias)
        self.bit = bit
        self.pwr_coef =  2**(bit - 1) 
        self.Round_w = RoundFn_LLSQ.apply
        self.Round_b = RoundFn_Bias.apply
        self.bias_flag = bias
        self.adc_list = adc_list
        #self.alpha_w = Variable(torch.rand( out_channels,1,1,1)).cuda()
        #self.alpha_w = Parameter(torch.rand( out_channels))
        #self.alpha_qfn = quan_fn_alpha()
        nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')
        if extern_init:
            param=list(init_model.parameters())
            self.weight=Parameter(param[0])
            if bias:
                self.bias=Parameter(param[1])
        if bit < 0:
            self.alpha_w = None
            self.init_state = 0
        else:
            self.alpha_w = Parameter(torch.rand( out_channels))
            self.register_buffer('init_state', torch.zeros(1))
        # self.init_state = 0

        self.sram_cim_conv = sram_cim_conv.apply
    
    def forward(self, conv_inp):
        x = conv_inp['act']
        alpha = conv_inp['alpha']
        
        if self.bit == 32:
            return F.conv2d(
                x, self.weight, self.bias, self.stride, self.padding,
                self.dilation, self.groups)
        else:
            w_reshape = self.weight.reshape([self.weight.shape[0], -1]).transpose(0, 1)
            if self.training and self.init_state == 0:            
                self.alpha_w.data.copy_(w_reshape.detach().abs().max(dim=0)[0] / self.pwr_coef)
                self.init_state.fill_(1)
               
            #assert not torch.isnan(x).any(), "Conv2d Input should not be 'nan'"
            #self.alpha_qfn(self.alpha_w)
            #if torch.isnan(self.alpha_w).any() or torch.isinf(self.alpha_w).any():
            #    assert not torch.isnan(wq).any(), self.alpha_w
            #    assert not torch.isinf(wq).any(), self.alpha_w
            wq =  self.Round_w(w_reshape, self.alpha_w, self.pwr_coef, self.bit)
            w_q = wq.transpose(0, 1).reshape(self.weight.shape)

            if self.bias_flag == True:
                LLSQ_b  = self.Round_b(self.bias, self.alpha_w, self.pwr_coef, self.bit)
            else:
                LLSQ_b = self.bias
            
            # assert not torch.isnan(self.weight).any(), "Weight should not be 'nan'"
            # if torch.isnan(wq).any() or torch.isinf(wq).any():
            #     print(self.alpha_w)
            #     assert not torch.isnan(wq).any(), "Conv2d Weights should not be 'nan'"
            #     assert not torch.isinf(wq).any(), "Conv2d Weights should not be 'nan'"
            outputs = self.sram_cim_conv(x, w_q, self.stride, self.padding, alpha, self.alpha_w, LLSQ_b, self.adc_list)
            return outputs

    def extra_repr(self):
        s_prefix = super(Conv2d_CIM_SRAM, self).extra_repr()
        if self.alpha_w is None:
            return '{}, fake'.format(s_prefix)
        return '{}'.format(s_prefix)
    
class Conv2d_CIM_SRAM_dw(nn.Conv2d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size, 
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1, 
                 bias=False,
                 bit=32,
                 extern_init=False,
                 init_model=nn.Sequential(),
                 adc_list=None):
        super(Conv2d_CIM_SRAM_dw, self).__init__(in_channels, out_channels, kernel_size, stride, padding,
                                              dilation, groups, bias)
        self.bit = bit
        self.pwr_coef =  2**(bit - 1) 
        self.Round_w = RoundFn_LLSQ.apply
        self.Round_b = RoundFn_Bias.apply
        self.bias_flag = bias
        self.adc_list = adc_list
        #self.alpha_w = Variable(torch.rand( out_channels,1,1,1)).cuda()
        #self.alpha_w = Parameter(torch.rand( out_channels))
        #self.alpha_qfn = quan_fn_alpha()
        nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')
        if extern_init:
            param=list(init_model.parameters())
            self.weight=Parameter(param[0])
            if bias:
                self.bias=Parameter(param[1])
        if bit < 0:
            self.alpha_w = None
            self.init_state = 0
        else:
            self.alpha_w = Parameter(torch.rand( out_channels))
            self.register_buffer('init_state', torch.zeros(1))
        # self.init_state = 0

        self.sram_cim_conv_dw = sram_cim_conv_dw.apply
    
    def forward(self, conv_inp):
        x = conv_inp['act']
        alpha = conv_inp['alpha']
        
        if self.bit == 32:
            return F.conv2d(
                x, self.weight, self.bias, self.stride, self.padding,
                self.dilation, self.groups)
        else:
            w_reshape = self.weight.reshape([self.weight.shape[0], -1]).transpose(0, 1)
            if self.training and self.init_state == 0:            
                self.alpha_w.data.copy_(w_reshape.detach().abs().max(dim=0)[0] / self.pwr_coef)
                self.init_state.fill_(1)
               
            #assert not torch.isnan(x).any(), "Conv2d Input should not be 'nan'"
            #self.alpha_qfn(self.alpha_w)
            #if torch.isnan(self.alpha_w).any() or torch.isinf(self.alpha_w).any():
            #    assert not torch.isnan(wq).any(), self.alpha_w
            #    assert not torch.isinf(wq).any(), self.alpha_w
            wq =  self.Round_w(w_reshape, self.alpha_w, self.pwr_coef, self.bit)
            w_q = wq.transpose(0, 1).reshape(self.weight.shape)

            if self.bias_flag == True:
                LLSQ_b  = self.Round_b(self.bias, self.alpha_w, self.pwr_coef, self.bit)
            else:
                LLSQ_b = self.bias
            
            # assert not torch.isnan(self.weight).any(), "Weight should not be 'nan'"
            # if torch.isnan(wq).any() or torch.isinf(wq).any():
            #     print(self.alpha_w)
            #     assert not torch.isnan(wq).any(), "Conv2d Weights should not be 'nan'"
            #     assert not torch.isinf(wq).any(), "Conv2d Weights should not be 'nan'"
            outputs = self.sram_cim_conv_dw(x, w_q, self.stride, self.padding, alpha, self.alpha_w, LLSQ_b, self.adc_list)
            return outputs

    def extra_repr(self):
        s_prefix = super(Conv2d_CIM_SRAM, self).extra_repr()
        if self.alpha_w is None:
            return '{}, fake'.format(s_prefix)
        return '{}'.format(s_prefix)
