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
                x_int = torch.round(input_unfold[:, :, i, :] / scale_alpha).int()   # 255 <- 2**8 - 1
                w_int = torch.round(torch.clamp(weight_unfold[i, :, :] / scale_alpha_w, min=-128, max=127)).int()   # [-128, 127]
                
                msb_x_int = x_int >> 4
                lsb_x_int = x_int & 15
        
                output_unfold += ( (CIM_MAC(msb_x_int, w_int, adc_list) << 4) + CIM_MAC(lsb_x_int, w_int, adc_list) )
                
            output_unfold = (output_unfold * scale_alpha_w).transpose(1, 2)                            # [B, K, OH*OW]
            output += torch.nn.functional.fold(output_unfold, (OH, OW), (1, 1))
        output = output * scale_alpha    # 8a8w

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

def w_nonideal(x):
    #SEED = 0
    #torch.manual_seed(SEED)
    #torch.cuda.manual_seed(SEED)

    map = torch.rand(x.shape)
    temp = x#.cpu()
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
    '''
    ##########################
    # Revise non ideal model #
    ##########################
    
    #map = torch.rand(x.shape)
    temp_ni = x#.cpu()
    
    x[(temp_ni == 0) & (map < 0.002)] += 1

    x[(temp_ni == 1) & (map < 0.001)] += 1

    x[(temp_ni == 2) & (map < 0.006)] += 1
    x[(temp_ni == 2) & (map >= 0.999)] -=1
    
    x[(temp_ni == 3) & (map < 0.004)] += 2
    x[(temp_ni == 3) & (map >= 0.004) & (map < 0.012)] += 1
    x[(temp_ni == 3) & (map >= 0.975)] -= 1
    
    x[(temp_ni == 4) & (map < 0.013)] += 1
    x[(temp_ni == 4) & (map >= 0.967) & (map < 0.984)] -= 1
    x[(temp_ni == 4) & (map >= 0.984)] -= 2

    x[(temp_ni == 5) & (map < 0.004)] += 2
    x[(temp_ni == 5) & (map >= 0.004) & (map < 0.019)] += 1
    x[(temp_ni == 5) & (map >= 0.984) & (map < 0.993)] -= 1
    x[(temp_ni == 5) & (map >= 0.993)] -= 2

    x[(temp_ni == 6) & (map < 0.01)] += 2
    x[(temp_ni == 6) & (map >= 0.01) & (map < 0.027)] += 1
    x[(temp_ni == 6) & (map >= 0.985) & (map < 0.997)] -= 1
    x[(temp_ni == 6) & (map >= 0.997)] -= 2

    x[(temp_ni == 7) & (map < 0.005)] += 2
    x[(temp_ni == 7) & (map >= 0.005) & (map < 0.023)] += 1
    x[(temp_ni == 7) & (map >= 0.985) & (map < 0.996)] -= 1
    x[(temp_ni == 7) & (map >= 0.996)] -= 2

    x[(temp_ni == 8) & (map < 0.003)] += 2
    x[(temp_ni == 8) & (map >= 0.003) & (map < 0.057)] += 1
    x[(temp_ni == 8) & (map >= 0.98) & (map < 0.999)] -= 1
    x[(temp_ni == 8) & (map >= 0.999)] -= 2

    x[(temp_ni == 9) & (map < 0.007)] += 2
    x[(temp_ni == 9) & (map >= 0.007) & (map < 0.057)] += 1
    x[(temp_ni == 9) & (map >= 0.986) & (map < 0.996)] -= 1
    x[(temp_ni == 9) & (map >= 0.996)] -= 2

    x[(temp_ni == 10) & (map < 0.001)] += 2
    x[(temp_ni == 10) & (map >= 0.001) & (map < 0.034)] += 1
    x[(temp_ni == 10) & (map >= 0.977) & (map < 0.999)] -= 1
    x[(temp_ni == 10) & (map >= 0.999)] -= 2

    x[(temp_ni == 11) & (map < 0.034)] += 1
    x[(temp_ni == 11) & (map >= 0.973) & (map < 0.995)] -=1
    x[(temp_ni == 11) & (map >= 0.995)] -= 2

    x[(temp_ni == 12) & (map < 0.011)] += 2
    x[(temp_ni == 12) & (map >= 0.011) & (map < 0.023)] += 1
    x[(temp_ni == 12) & (map >= 0.974) & (map < 0.996)] -= 1
    x[(temp_ni == 12) & (map >= 0.996)] -= 2

    x[(temp_ni == 13) & (map < 0.007)] += 2
    x[(temp_ni == 13) & (map >= 0.007) & (map < 0.021)] += 1
    x[(temp_ni == 13) & (map >= 0.987) & (map < 0.996)] -= 1
    x[(temp_ni == 13) & (map >= 0.996)] -= 2

    x[(temp_ni == 14) & (map < 0.012)] += 2
    x[(temp_ni == 14) & (map >= 0.012) & (map < 0.051)] += 1
    x[(temp_ni == 14) & (map >= 0.989) & (map < 0.998)] -= 1
    x[(temp_ni == 14) & (map >= 0.998)] -= 2

    x[(temp_ni == 15) & (map < 0.02)] += 2
    x[(temp_ni == 15) & (map >= 0.02) & (map < 0.078)] += 1
    x[(temp_ni == 15) & (map >= 0.997)] -= 1

    x[(temp_ni == 16) & (map < 0.02)] += 2
    x[(temp_ni == 16) & (map >= 0.02) & (map < 0.102)] += 1
    x[(temp_ni == 16) & (map >= 0.989)] -= 1

    x[(temp_ni == 17) & (map < 0.002)] += 3
    x[(temp_ni == 17) & (map >= 0.002) & (map < 0.013)] += 2
    x[(temp_ni == 17) & (map >= 0.013) & (map < 0.136)] += 1
    x[(temp_ni == 17) & (map >= 0.948)] -= 1

    x[(temp_ni == 18) & (map < 0.001)] += 3
    x[(temp_ni == 18) & (map >= 0.001) & (map < 0.018)] += 2
    x[(temp_ni == 18) & (map >= 0.018) & (map < 0.151)] += 1
    x[(temp_ni == 18) & (map >= 0.95) & (map < 0.989)] -= 1
    x[(temp_ni == 18) & (map >= 0.989)] -= 2

    x[(temp_ni == 19) & (map < 0.004)] += 3
    x[(temp_ni == 19) & (map >= 0.004) & (map < 0.018)] += 2
    x[(temp_ni == 19) & (map >= 0.018) & (map < 0.147)] += 1
    x[(temp_ni == 19) & (map >= 0.899) & (map < 0.977)] -= 1
    x[(temp_ni == 19) & (map >= 0.977)] -= 2

    x[(temp_ni == 20) & (map < 0.004)] += 3
    x[(temp_ni == 20) & (map >= 0.004) & (map < 0.035)] += 2
    x[(temp_ni == 20) & (map >= 0.035) & (map < 0.147)] += 1
    x[(temp_ni == 20) & (map >= 0.876) & (map < 0.984)] -= 1
    x[(temp_ni == 20) & (map >= 0.984)] -= 2

    x[(temp_ni == 21) & (map < 0.003)] += 3
    x[(temp_ni == 21) & (map >= 0.003) & (map < 0.031)] += 2
    x[(temp_ni == 21) & (map >= 0.031) & (map < 0.15)] += 1
    x[(temp_ni == 21) & (map >= 0.862) & (map < 0.976)] -= 1
    x[(temp_ni == 21) & (map >= 0.976)] -= 2

    x[(temp_ni == 22) & (map < 0.006)] += 3
    x[(temp_ni == 22) & (map >= 0.006) & (map < 0.028)] += 2
    x[(temp_ni == 22) & (map >= 0.028) & (map < 0.162)] += 1
    x[(temp_ni == 22) & (map >= 0.868) & (map < 0.985)] -= 1
    x[(temp_ni == 22) & (map >= 0.985) & (map < 0.992)] -= 2
    x[(temp_ni == 22) & (map >= 0.992)] -= 3

    x[(temp_ni == 23) & (map < 0.011)] += 3
    x[(temp_ni == 23) & (map >= 0.011) & (map < 0.025)] += 2
    x[(temp_ni == 23) & (map >= 0.025) & (map < 0.109)] += 1
    x[(temp_ni == 23) & (map >= 0.896) & (map < 0.977)] -= 1
    x[(temp_ni == 23) & (map >= 0.977) & (map < 0.994)] -= 2
    x[(temp_ni == 23) & (map >= 0.994)] -= 3

    x[(temp_ni == 24) & (map < 0.012)] += 2
    x[(temp_ni == 24) & (map >= 0.012) & (map < 0.135)] += 1
    x[(temp_ni == 24) & (map >= 0.917) & (map < 0.977)] -= 1
    x[(temp_ni == 24) & (map >= 0.977) & (map < 0.997)] -= 2
    x[(temp_ni == 24) & (map >= 0.997)] -= 3

    x[(temp_ni == 25) & (map < 0.008)] += 2
    x[(temp_ni == 25) & (map >= 0.008) & (map < 0.151)] += 1
    x[(temp_ni == 25) & (map >= 0.9) & (map < 0.977)] -= 1
    x[(temp_ni == 25) & (map >= 0.977) & (map < 0.99)] -= 2
    x[(temp_ni == 25) & (map >= 0.99)] -= 3

    x[(temp_ni == 26) & (map < 0.016)] += 2
    x[(temp_ni == 26) & (map >= 0.016) & (map < 0.043)] += 1
    x[(temp_ni == 26) & (map >= 0.876) & (map < 0.976)] -= 1
    x[(temp_ni == 26) & (map >= 0.976)] -= 2

    x[(temp_ni == 27) & (map < 0.011)] += 2
    x[(temp_ni == 27) & (map >= 0.011) & (map < 0.032)] += 1
    x[(temp_ni == 27) & (map >= 0.935) & (map < 0.983)] -= 1
    x[(temp_ni == 27) & (map >= 0.983)] -= 2

    x[(temp_ni == 28) & (map < 0.021)] += 1
    x[(temp_ni == 28) & (map >= 0.961) & (map < 0.987)] -= 1
    x[(temp_ni == 28) & (map >= 0.987)] -= 2

    x[(temp_ni == 29) & (map < 0.004)] += 2
    x[(temp_ni == 29) & (map >= 0.004) & (map < 0.022)] += 1
    x[(temp_ni == 29) & (map >= 0.976) & (map < 0.991)] -= 1
    x[(temp_ni == 29) & (map >= 0.991)] -= 2

    x[(temp_ni == 30) & (map < 0.008)] += 1
    x[(temp_ni == 30) & (map >= 0.987) & (map < 0.995)] -= 1
    x[(temp_ni == 30) & (map >= 0.995)] -= 2

    x[(temp_ni == 31) & (map >= 0.987)] -= 1
    '''
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
    #SEED = 0
    #torch.manual_seed(SEED)
    #torch.cuda.manual_seed(SEED)

    map = torch.rand(x.shape)
    temp = x#.cpu()
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
    '''
    ##########################
    # Revise non ideal model #
    ##########################
    
    #map = torch.rand(x.shape)
    temp_ni = x#.cpu()
    
    x[(temp_ni == 0) & (map < 0.001)] += 1

    x[(temp_ni == 1) & (map < 0.001)] += 1

    x[(temp_ni == 2) & (map < 0.006)] += 1
    x[(temp_ni == 2) & (map >= 0.999)] -= 1
    
    x[(temp_ni == 3) & (map < 0.006)] += 1
    x[(temp_ni == 3) & (map >= 0.976)] -= 1

    x[(temp_ni == 4) & (map < 0.012)] += 1
    x[(temp_ni == 4) & (map >= 0.989)] -= 1
    
    x[(temp_ni == 5) & (map < 0.004)] += 1
    x[(temp_ni == 5) & (map >= 0.999)] -= 1
    
    x[(temp_ni == 6) & (map < 0.028)] += 1
    x[(temp_ni == 6) & (map >= 0.996)] -= 1
    
    x[(temp_ni == 7) & (map < 0.028)] += 1
    x[(temp_ni == 7) & (map >= 0.995)] -= 1

    x[(temp_ni == 8) & (map < 0.05)] += 1
    x[(temp_ni == 8) & (map >= 0.998)] -= 1

    x[(temp_ni == 9) & (map < 0.058)] += 1
    x[(temp_ni == 9) & (map >= 0.998)] -= 1

    x[(temp_ni == 10) & (map < 0.043)] += 1
    x[(temp_ni == 10) & (map >= 0.978)] -= 1
    
    x[(temp_ni == 11) & (map < 0.031)] += 1
    x[(temp_ni == 11) & (map >= 0.998)] -= 1
    
    x[(temp_ni == 12) & (map < 0.003)] += 1
    x[(temp_ni == 12) & (map >= 0.987)] -= 1

    x[(temp_ni == 13) & (map < 0.021)] += 1
    x[(temp_ni == 13) & (map >= 0.997)] -= 1

    x[(temp_ni == 14) & (map < 0.042)] += 1
    x[(temp_ni == 14) & (map >= 0.997)] -= 1
    
    x[(temp_ni == 15) & (map < 0.063)] += 1
    x[(temp_ni == 15) & (map >= 0.997)] -= 1

    x[(temp_ni == 16) & (map < 0.089)] += 1
    x[(temp_ni == 16) & (map >= 0.983)] -= 1
    
    x[(temp_ni == 17) & (map < 0.007)] += 2
    x[(temp_ni == 17) & (map >= 0.007) & (map < 0.127)] += 1
    x[(temp_ni == 17) & (map >= 0.959)] -= 1
    
    x[(temp_ni == 18) & (map < 0.002)] += 2
    x[(temp_ni == 18) & (map >= 0.002) & (map < 0.163)] += 1
    x[(temp_ni == 18) & (map >= 0.938) & (map < 0.988)] -= 1
    x[(temp_ni == 18) & (map >= 0.988)] -= 2
    
    x[(temp_ni == 19) & (map < 0.002)] += 2
    x[(temp_ni == 19) & (map >= 0.002) & (map < 0.161)] += 1
    x[(temp_ni == 19) & (map >= 0.925)] -= 1
    
    x[(temp_ni == 20) & (map < 0.021)] += 2
    x[(temp_ni == 20) & (map >= 0.021) & (map < 0.154)] += 1
    x[(temp_ni == 20) & (map >= 0.874) & (map < 0.986)] -= 1
    x[(temp_ni == 20) & (map >= 0.986)] -= 2

    x[(temp_ni == 21) & (map < 0.013)] += 2
    x[(temp_ni == 21) & (map >= 0.013) & (map < 0.157)] += 1
    x[(temp_ni == 21) & (map >= 0.884) & (map < 0.992)] -= 1
    x[(temp_ni == 21) & (map >= 0.992)] -= 2
    
    x[(temp_ni == 22) & (map < 0.153)] += 1
    x[(temp_ni == 22) & (map >= 0.891)] -= 1
    
    x[(temp_ni == 23) & (map < 0.01)] += 2
    x[(temp_ni == 23) & (map >= 0.01) & (map < 0.107)] += 1
    x[(temp_ni == 23) & (map >= 0.899) & (map < 0.983)] -= 1
    x[(temp_ni == 23) & (map >= 0.983)] -= 2

    x[(temp_ni == 24) & (map < 0.011)] += 2
    x[(temp_ni == 24) & (map >= 0.011) & (map < 0.149)] += 1
    x[(temp_ni == 24) & (map >= 0.936) & (map < 0.987)] -= 1
    x[(temp_ni == 24) & (map >= 0.987)] -= 2

    x[(temp_ni == 25) & (map < 0.001)] += 2
    x[(temp_ni == 25) & (map >= 0.001) & (map < 0.135)] += 1
    x[(temp_ni == 25) & (map >= 0.92)] -= 1
    
    x[(temp_ni == 26) & (map < 0.033)] += 1
    x[(temp_ni == 26) & (map >= 0.897)] -= 1
    
    x[(temp_ni == 27) & (map < 0.004)] += 1
    x[(temp_ni == 27) & (map >= 0.959)] -= 1
    
    x[(temp_ni == 28) & (map < 0.006)] += 1
    x[(temp_ni == 28) & (map >= 0.988)] -= 1
    
    x[(temp_ni == 29) & (map < 0.014)] += 1
    x[(temp_ni == 29) & (map >= 0.986)] -= 1
    
    x[(temp_ni == 30) & (map < 0.009)] += 1
    x[(temp_ni == 30) & (map >= 0.996)] -= 1
    
    x[(temp_ni == 31) & (map >= 0.996)] -= 1
    '''
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
    #writeout_dis('w_comb', x, adc_list)
    #output = w_nonideal(x) * 5 + 2
    output = x * 5 + 2
    
    return output

def input2_qsa(x, adc_list):
    '''
    return x.clamp_max(31)
    '''
    # way-2
    x = x.clamp_max(31)
    #writeout_dis('wo_comb', x, adc_list)
    #output = wo_nonideal(x)
    output = x
    
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

def writeout_dis(name, convX, adc_list):
    filepath = './inception_adc_distribution/' + name + '.dat'
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
    conv3_msb = input2_qsa(conv3_msb, adc_list[0])
    conv3_lsb = input2_qsa(conv3_lsb, adc_list[0])
    conv2_msb = input2_qsa(conv2_msb, adc_list[0])
    conv2_lsb = input2_qsa(conv2_lsb, adc_list[0])
    
    '''
    conv3_msb = ADC.apply(conv3_msb)
    conv3_lsb = ADC.apply(conv3_lsb)
    conv2_msb = ADC.apply(conv2_msb)
    conv2_lsb = ADC.apply(conv2_lsb)
    conv1_msb = ADC.apply(conv1_msb)
    conv1_lsb = ADC.apply(conv1_lsb)
    conv0_msb = ADC.apply(conv0_msb)
    conv0_lsb = ADC.apply(conv0_lsb)
    
    conv5 = ADC_comb.apply(conv5)
    conv4 = ADC_comb.apply(conv4)
    #conv3 = ADC_comb.apply(conv3)
    #conv2 = ADC_comb.apply(conv2)
    #conv1 = ADC_comb.apply(conv1)
    #conv0 = ADC_comb.apply(conv0)
    conv3 = input4_qsa(conv3)
    conv2 = input4_qsa(conv2)
    '''
    conv1 = input4_qsa(conv1, adc_list[1])
    conv0 = input4_qsa(conv0, adc_list[1])
    
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
                adc_list=None,):
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
