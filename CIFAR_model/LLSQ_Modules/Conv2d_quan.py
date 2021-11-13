# %%
from numpy.core.fromnumeric import reshape
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
    def forward(ctx, input, weight, stride, padding, scale_alpha, scale_alpha_w, bias=None, comb_list=None):
        ctx.save_for_backward(input, weight, bias)
        ctx.stride = stride
        ctx.padding = padding
        ctx.device = input.device
        
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
        output = torch.zeros((B, K, OH, OW), device=ctx.device)                                        # [B, K, OH, OW]
        
        for gp in range(GPS):
            input_unfold = torch.nn.functional.unfold(input_slices[:, gp, :, :, :], # [B, alpha*KH*KW, OH*OW]
                                                    kernel_size=(KH, KW), 
                                                    stride=stride, 
                                                    padding=padding)
            input_unfold = input_unfold.transpose(1, 2)                             # [B, OH*OW, alpha*KH*KW]
            input_unfold = input_unfold.view(B, OH*OW, KH*KW, alpha)                # [B, OH*OW, KH*KW, alpha]
            
            weight_unfold = weight_slices[:, gp, :, :, :].view(K, -1).t()           # [alpha*KH*KW, K]
            weight_unfold = weight_unfold.view(KH*KW, alpha, K)                     # [KH*KW, alpha, K]
                
            output_unfold = torch.zeros((B, OH*OW, K), device=ctx.device)                              # [B, OH*OW, K]
            
            for i in range(KH*KW):
                # 8a8w
                # FP --> Int
                x_int = torch.round(input_unfold[:, :, i, :] / scale_alpha).int()                                   # [B, OH*OW, alpha]      255 <- 2**8 - 1
                w_int = torch.round(torch.clamp(weight_unfold[i, :, :] / scale_alpha_w, min=-128, max=127)).int()   # [alpha, K]             128 <- 2**(8-1)
                
                msb_x_int = x_int >> 4
                lsb_x_int = x_int & 15
                
                output_unfold += ( (CIM_MAC(msb_x_int, w_int, comb_list, ctx.device) << 4) + CIM_MAC(lsb_x_int, w_int, comb_list, ctx.device) )
                
            output_unfold = (output_unfold * scale_alpha_w).transpose(1, 2)                            # [B, K, OH*OW]
            output += torch.nn.functional.fold(output_unfold, (OH, OW), (1, 1))
        output = output * scale_alpha        # 8a8w

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

def w_nonideal(x):
    map = torch.rand(x.shape)
    temp = x.cpu()
    
    x[(temp == 0) & (map >= 0.99)] += 1
    
    x[(temp == 1) & (map < 0.01)] -= 1
    x[(temp == 1) & (map >= 0.99)] += 1
    
    x[(temp == 2) & (map < 0.02)] -= 1
    x[(temp == 2) & (map >= 0.99)] += 1
    
    x[(temp == 3) & (map < 0.02)] -= 1
    x[(temp == 3) & (map >= 0.98)] += 1
    
    x[(temp == 4) & (map < 0.02)] -= 1
    x[(temp == 4) & (map >= 0.98)] += 1
    
    x[(temp == 5) & (map < 0.02)] -= 1
    x[(temp == 5) & (map >= 0.98)] += 1
    
    x[(temp == 6) & (map < 0.01)] -= 2
    x[(temp == 6) & (map >= 0.01) & (map < 0.03)] -= 1
    x[(temp == 6) & (map >= 0.98)] += 1
    
    x[(temp == 7) & (map < 0.01)] -= 2
    x[(temp == 7) & (map >= 0.01) & (map < 0.03)] -= 1
    x[(temp == 7) & (map >= 0.97) & (map < 0.99)] += 1
    x[(temp == 7) & (map >= 0.99)] += 2

    x[(temp == 8) & (map < 0.01)] -= 2
    x[(temp == 8) & (map >= 0.01) & (map < 0.03)] -= 1
    x[(temp == 8) & (map >= 0.97) & (map < 0.99)] += 1
    x[(temp == 8) & (map >= 0.99)] += 2

    x[(temp == 9) & (map < 0.01)] -= 3
    x[(temp == 9) & (map >= 0.01) & (map < 0.02)] -= 2
    x[(temp == 9) & (map >= 0.02) & (map < 0.04)] -= 1
    x[(temp == 9) & (map >= 0.97) & (map < 0.99)] += 1
    x[(temp == 9) & (map >= 0.99)] += 2

    x[(temp == 10) & (map < 0.01)] -= 3
    x[(temp == 10) & (map >= 0.01) & (map < 0.02)] -= 2
    x[(temp == 10) & (map >= 0.02) & (map < 0.04)] -= 1
    x[(temp == 10) & (map >= 0.97) & (map < 0.99)] += 1
    x[(temp == 10) & (map >= 0.99)] += 2

    x[(temp == 11) & (map < 0.01)] -= 3
    x[(temp == 11) & (map >= 0.01) & (map < 0.02)] -= 2
    x[(temp == 11) & (map >= 0.02) & (map < 0.04)] -= 1
    x[(temp == 11) & (map >= 0.97) & (map < 0.99)] += 1
    x[(temp == 11) & (map >= 0.99)] += 2

    x[(temp == 12) & (map < 0.01)] -= 2
    x[(temp == 12) & (map >= 0.01) & (map < 0.04)] -= 1
    x[(temp == 12) & (map >= 0.96) & (map < 0.98)] += 1
    x[(temp == 12) & (map >= 0.98) & (map < 0.99)] += 2
    x[(temp == 12) & (map >= 0.99)] += 3

    x[(temp == 13) & (map < 0.01)] -= 3
    x[(temp == 13) & (map >= 0.01) & (map < 0.02)] -= 2
    x[(temp == 13) & (map >= 0.02) & (map < 0.05)] -= 1
    x[(temp == 13) & (map >= 0.97) & (map < 0.99)] += 1
    x[(temp == 13) & (map >= 0.99)] += 2

    x[(temp == 14) & (map < 0.01)] -= 3
    x[(temp == 14) & (map >= 0.01) & (map < 0.02)] -= 2
    x[(temp == 14) & (map >= 0.02) & (map < 0.05)] -= 1
    x[(temp == 14) & (map >= 0.96) & (map < 0.98)] += 1
    x[(temp == 14) & (map >= 0.98) & (map < 0.99)] += 2
    x[(temp == 14) & (map >= 0.99)] += 3

    x[(temp == 15) & (map < 0.01)] -= 3
    x[(temp == 15) & (map >= 0.01) & (map < 0.02)] -= 2
    x[(temp == 15) & (map >= 0.02) & (map < 0.05)] -= 1
    x[(temp == 15) & (map >= 0.95) & (map < 0.97)] += 1
    x[(temp == 15) & (map >= 0.97) & (map < 0.98)] += 2
    x[(temp == 15) & (map >= 0.98) & (map < 0.99)] += 3
    x[(temp == 15) & (map >= 0.99)] += 4

    x[(temp == 16) & (map < 0.01)] -= 4
    x[(temp == 16) & (map >= 0.01) & (map < 0.02)] -= 3
    x[(temp == 16) & (map >= 0.02) & (map < 0.03)] -= 2
    x[(temp == 16) & (map >= 0.03) & (map < 0.05)] -= 1
    x[(temp == 16) & (map >= 0.95) & (map < 0.97)] += 1
    x[(temp == 16) & (map >= 0.97) & (map < 0.99)] += 2
    x[(temp == 16) & (map >= 0.99)] += 3

    x[(temp == 17) & (map < 0.01)] -= 3
    x[(temp == 17) & (map >= 0.01) & (map < 0.02)] -= 2
    x[(temp == 17) & (map >= 0.02) & (map < 0.05)] -= 1
    x[(temp == 17) & (map >= 0.95) & (map < 0.97)] += 1
    x[(temp == 17) & (map >= 0.97) & (map < 0.99)] += 2
    x[(temp == 17) & (map >= 0.99)] += 3

    x[(temp == 18) & (map < 0.01)] -= 4
    x[(temp == 18) & (map >= 0.01) & (map < 0.02)] -= 3
    x[(temp == 18) & (map >= 0.02) & (map < 0.03)] -= 2
    x[(temp == 18) & (map >= 0.03) & (map < 0.06)] -= 1
    x[(temp == 18) & (map >= 0.95) & (map < 0.98)] += 1
    x[(temp == 18) & (map >= 0.98) & (map < 0.99)] += 2
    x[(temp == 18) & (map >= 0.99)] += 3

    x[(temp == 19) & (map < 0.01)] -= 3
    x[(temp == 19) & (map >= 0.01) & (map < 0.02)] -= 2
    x[(temp == 19) & (map >= 0.02) & (map < 0.05)] -= 1
    x[(temp == 19) & (map >= 0.94) & (map < 0.97)] += 1
    x[(temp == 19) & (map >= 0.97) & (map < 0.98)] += 2
    x[(temp == 19) & (map >= 0.98) & (map < 0.99)] += 3
    x[(temp == 19) & (map >= 0.99)] += 4

    x[(temp == 20) & (map < 0.01)] -= 4
    x[(temp == 20) & (map >= 0.01) & (map < 0.02)] -= 3
    x[(temp == 20) & (map >= 0.02) & (map < 0.03)] -= 2
    x[(temp == 20) & (map >= 0.03) & (map < 0.06)] -= 1
    x[(temp == 20) & (map >= 0.94) & (map < 0.97)] += 1
    x[(temp == 20) & (map >= 0.97) & (map < 0.98)] += 2
    x[(temp == 20) & (map >= 0.98) & (map < 0.99)] += 3
    x[(temp == 20) & (map >= 0.99)] += 4

    x[(temp == 21) & (map < 0.01)] -= 4
    x[(temp == 21) & (map >= 0.01) & (map < 0.02)] -= 3
    x[(temp == 21) & (map >= 0.02) & (map < 0.04)] -= 2
    x[(temp == 21) & (map >= 0.04) & (map < 0.08)] -= 1
    x[(temp == 21) & (map >= 0.94) & (map < 0.97)] += 1
    x[(temp == 21) & (map >= 0.97) & (map < 0.98)] += 2
    x[(temp == 21) & (map >= 0.98) & (map < 0.99)] += 3
    x[(temp == 21) & (map >= 0.99)] += 4

    x[(temp == 22) & (map < 0.01)] -= 4
    x[(temp == 22) & (map >= 0.01) & (map < 0.03)] -= 3
    x[(temp == 22) & (map >= 0.03) & (map < 0.05)] -= 2
    x[(temp == 22) & (map >= 0.05) & (map < 0.09)] -= 1
    x[(temp == 22) & (map >= 0.92) & (map < 0.96)] += 1
    x[(temp == 22) & (map >= 0.96) & (map < 0.98)] += 2
    x[(temp == 22) & (map >= 0.98) & (map < 0.99)] += 3
    x[(temp == 22) & (map >= 0.99)] += 4

    x[(temp == 23) & (map < 0.01)] -= 5
    x[(temp == 23) & (map >= 0.01) & (map < 0.02)] -= 4
    x[(temp == 23) & (map >= 0.02) & (map < 0.03)] -= 3
    x[(temp == 23) & (map >= 0.03) & (map < 0.05)] -= 2
    x[(temp == 23) & (map >= 0.05) & (map < 0.09)] -= 1
    x[(temp == 23) & (map >= 0.90) & (map < 0.95)] += 1
    x[(temp == 23) & (map >= 0.95) & (map < 0.97)] += 2
    x[(temp == 23) & (map >= 0.97) & (map < 0.98)] += 3
    x[(temp == 23) & (map >= 0.98) & (map < 0.99)] += 4
    x[(temp == 23) & (map >= 0.99)] += 5

    x[(temp == 24) & (map < 0.01)] -= 4
    x[(temp == 24) & (map >= 0.01) & (map < 0.02)] -= 3
    x[(temp == 24) & (map >= 0.02) & (map < 0.05)] -= 2
    x[(temp == 24) & (map >= 0.05) & (map < 0.11)] -= 1
    x[(temp == 24) & (map >= 0.89) & (map < 0.94)] += 1
    x[(temp == 24) & (map >= 0.94) & (map < 0.97)] += 2
    x[(temp == 24) & (map >= 0.97) & (map < 0.98)] += 3
    x[(temp == 24) & (map >= 0.98) & (map < 0.99)] += 4
    x[(temp == 24) & (map >= 0.99)] += 5

    x[(temp == 25) & (map < 0.01)] -= 5
    x[(temp == 25) & (map >= 0.01) & (map < 0.02)] -= 4
    x[(temp == 25) & (map >= 0.02) & (map < 0.03)] -= 3
    x[(temp == 25) & (map >= 0.03) & (map < 0.05)] -= 2
    x[(temp == 25) & (map >= 0.05) & (map < 0.09)] -= 1
    x[(temp == 25) & (map >= 0.90) & (map < 0.95)] += 1
    x[(temp == 25) & (map >= 0.95) & (map < 0.97)] += 2
    x[(temp == 25) & (map >= 0.97) & (map < 0.98)] += 3
    x[(temp == 25) & (map >= 0.98) & (map < 0.99)] += 4
    x[(temp == 25) & (map >= 0.99)] += 5

    x[(temp == 26) & (map < 0.01)] -= 4
    x[(temp == 26) & (map >= 0.01) & (map < 0.03)] -= 3
    x[(temp == 26) & (map >= 0.03) & (map < 0.05)] -= 2
    x[(temp == 26) & (map >= 0.05) & (map < 0.08)] -= 1
    x[(temp == 26) & (map >= 0.91) & (map < 0.95)] += 1
    x[(temp == 26) & (map >= 0.95) & (map < 0.98)] += 2
    x[(temp == 26) & (map >= 0.98) & (map < 0.99)] += 3
    x[(temp == 26) & (map >= 0.99)] += 4

    x[(temp == 27) & (map < 0.01)] -= 4
    x[(temp == 27) & (map >= 0.01) & (map < 0.02)] -= 3
    x[(temp == 27) & (map >= 0.02) & (map < 0.03)] -= 2
    x[(temp == 27) & (map >= 0.03) & (map < 0.06)] -= 1
    x[(temp == 27) & (map >= 0.92) & (map < 0.96)] += 1
    x[(temp == 27) & (map >= 0.96) & (map < 0.98)] += 2
    x[(temp == 27) & (map >= 0.98) & (map < 0.99)] += 3
    x[(temp == 27) & (map >= 0.99)] += 4

    x[(temp == 28) & (map < 0.01)] -= 3
    x[(temp == 28) & (map >= 0.01) & (map < 0.02)] -= 2
    x[(temp == 28) & (map >= 0.02) & (map < 0.04)] -= 1
    x[(temp == 28) & (map >= 0.93) & (map < 0.96)] += 1
    x[(temp == 28) & (map >= 0.96) & (map < 0.98)] += 2
    x[(temp == 28) & (map >= 0.98)] += 3

    x[(temp == 29) & (map < 0.01)] -= 3
    x[(temp == 29) & (map >= 0.01) & (map < 0.02)] -= 2
    x[(temp == 29) & (map >= 0.02) & (map < 0.04)] -= 1
    x[(temp == 29) & (map >= 0.93) & (map < 0.96)] += 1
    x[(temp == 29) & (map >= 0.96)] += 2

    x[(temp == 30) & (map < 0.02)] -= 2
    x[(temp == 30) & (map >= 0.02) & (map < 0.04)] -= 1
    x[(temp == 30) & (map >= 0.94)] += 1

    x[(temp == 31) & (map < 0.03)] -= 1

    return x
    
def wo_nonideal(x):
    map = torch.rand(x.shape)
    temp = x.cpu()
    
    x[(temp == 0) & (map >= 0.99)] += 1
    
    x[(temp == 1) & (map >= 0.99)] += 1
    
    x[(temp == 2) & (map < 0.01)] -= 1
    x[(temp == 2) & (map >= 0.99)] += 1
    
    x[(temp == 3) & (map < 0.02)] -= 1
    x[(temp == 3) & (map >= 0.99)] += 1
    
    x[(temp == 4) & (map < 0.02)] -= 1
    x[(temp == 4) & (map >= 0.99)] += 1
    
    x[(temp == 5) & (map < 0.01)] -= 1
    x[(temp == 5) & (map >= 0.98)] += 1
    
    x[(temp == 6) & (map < 0.02)] -= 1
    x[(temp == 6) & (map >= 0.98)] += 1
    
    x[(temp == 7) & (map < 0.01)] -= 2
    x[(temp == 7) & (map >= 0.01) & (map < 0.02)] -= 1
    x[(temp == 7) & (map >= 0.98)] += 1

    x[(temp == 8) & (map < 0.02)] -= 1
    x[(temp == 8) & (map >= 0.98)] += 1

    x[(temp == 9) & (map < 0.01)] -= 2
    x[(temp == 9) & (map >= 0.01) & (map < 0.03)] -= 1
    x[(temp == 9) & (map >= 0.98) & (map < 0.99)] += 1
    x[(temp == 9) & (map >= 0.99)] += 2

    x[(temp == 10) & (map < 0.01)] -= 2
    x[(temp == 10) & (map >= 0.01) & (map < 0.03)] -= 1
    x[(temp == 10) & (map >= 0.98) & (map < 0.99)] += 1
    x[(temp == 10) & (map >= 0.99)] += 2

    x[(temp == 11) & (map < 0.01)] -= 2
    x[(temp == 11) & (map >= 0.01) & (map < 0.03)] -= 1
    x[(temp == 11) & (map >= 0.98) & (map < 0.99)] += 1
    x[(temp == 11) & (map >= 0.99)] += 2

    x[(temp == 12) & (map < 0.01)] -= 2
    x[(temp == 12) & (map >= 0.01) & (map < 0.03)] -= 1
    x[(temp == 12) & (map >= 0.97) & (map < 0.98)] += 1
    x[(temp == 12) & (map >= 0.98) & (map < 0.99)] += 2
    x[(temp == 12) & (map >= 0.99)] += 3

    x[(temp == 13) & (map < 0.01)] -= 3
    x[(temp == 13) & (map >= 0.01) & (map < 0.02)] -= 2
    x[(temp == 13) & (map >= 0.02) & (map < 0.03)] -= 1
    x[(temp == 13) & (map >= 0.97) & (map < 0.98)] += 1
    x[(temp == 13) & (map >= 0.98) & (map < 0.99)] += 2
    x[(temp == 13) & (map >= 0.99)] += 3

    x[(temp == 14) & (map < 0.01)] -= 2
    x[(temp == 14) & (map >= 0.01) & (map < 0.03)] -= 1
    x[(temp == 14) & (map >= 0.97) & (map < 0.98)] += 1
    x[(temp == 14) & (map >= 0.98) & (map < 0.99)] += 2
    x[(temp == 14) & (map >= 0.99)] += 3

    x[(temp == 15) & (map < 0.01)] -= 3
    x[(temp == 15) & (map >= 0.01) & (map < 0.02)] -= 2
    x[(temp == 15) & (map >= 0.02) & (map < 0.04)] -= 1
    x[(temp == 15) & (map >= 0.97) & (map < 0.99)] += 1
    x[(temp == 15) & (map >= 0.99)] += 2

    x[(temp == 16) & (map < 0.01)] -= 3
    x[(temp == 16) & (map >= 0.01) & (map < 0.02)] -= 2
    x[(temp == 16) & (map >= 0.02) & (map < 0.04)] -= 1
    x[(temp == 16) & (map >= 0.97) & (map < 0.98)] += 1
    x[(temp == 16) & (map >= 0.98) & (map < 0.99)] += 2
    x[(temp == 16) & (map >= 0.99)] += 3

    x[(temp == 17) & (map < 0.01)] -= 3
    x[(temp == 17) & (map >= 0.01) & (map < 0.02)] -= 2
    x[(temp == 17) & (map >= 0.02) & (map < 0.04)] -= 1
    x[(temp == 17) & (map >= 0.96) & (map < 0.98)] += 1
    x[(temp == 17) & (map >= 0.98) & (map < 0.99)] += 2
    x[(temp == 17) & (map >= 0.99)] += 3

    x[(temp == 18) & (map < 0.01)] -= 4
    x[(temp == 18) & (map >= 0.01) & (map < 0.02)] -= 3
    x[(temp == 18) & (map >= 0.02) & (map < 0.03)] -= 2
    x[(temp == 18) & (map >= 0.03) & (map < 0.04)] -= 1
    x[(temp == 18) & (map >= 0.96) & (map < 0.98)] += 1
    x[(temp == 18) & (map >= 0.98) & (map < 0.99)] += 2
    x[(temp == 18) & (map >= 0.99)] += 3

    x[(temp == 19) & (map < 0.01)] -= 3
    x[(temp == 19) & (map >= 0.01) & (map < 0.02)] -= 2
    x[(temp == 19) & (map >= 0.02) & (map < 0.04)] -= 1
    x[(temp == 19) & (map >= 0.96) & (map < 0.98)] += 1
    x[(temp == 19) & (map >= 0.98) & (map < 0.99)] += 2
    x[(temp == 19) & (map >= 0.99)] += 3

    x[(temp == 20) & (map < 0.01)] -= 3
    x[(temp == 20) & (map >= 0.01) & (map < 0.02)] -= 2
    x[(temp == 20) & (map >= 0.02) & (map < 0.05)] -= 1
    x[(temp == 20) & (map >= 0.95) & (map < 0.97)] += 1
    x[(temp == 20) & (map >= 0.97) & (map < 0.99)] += 2
    x[(temp == 20) & (map >= 0.99)] += 3

    x[(temp == 21) & (map < 0.01)] -= 3
    x[(temp == 21) & (map >= 0.01) & (map < 0.03)] -= 2
    x[(temp == 21) & (map >= 0.03) & (map < 0.06)] -= 1
    x[(temp == 21) & (map >= 0.95) & (map < 0.97)] += 1
    x[(temp == 21) & (map >= 0.97) & (map < 0.98)] += 2
    x[(temp == 21) & (map >= 0.98) & (map < 0.99)] += 3
    x[(temp == 21) & (map >= 0.99)] += 4

    x[(temp == 22) & (map < 0.01)] -= 3
    x[(temp == 22) & (map >= 0.01) & (map < 0.03)] -= 2
    x[(temp == 22) & (map >= 0.03) & (map < 0.06)] -= 1
    x[(temp == 22) & (map >= 0.94) & (map < 0.96)] += 1
    x[(temp == 22) & (map >= 0.96) & (map < 0.98)] += 2
    x[(temp == 22) & (map >= 0.98) & (map < 0.99)] += 3
    x[(temp == 22) & (map >= 0.99)] += 4

    x[(temp == 23) & (map < 0.01)] -= 3
    x[(temp == 23) & (map >= 0.01) & (map < 0.03)] -= 2
    x[(temp == 23) & (map >= 0.03) & (map < 0.07)] -= 1
    x[(temp == 23) & (map >= 0.94) & (map < 0.97)] += 1
    x[(temp == 23) & (map >= 0.97) & (map < 0.99)] += 2
    x[(temp == 23) & (map >= 0.99)] += 3

    x[(temp == 24) & (map < 0.01)] -= 4
    x[(temp == 24) & (map >= 0.01) & (map < 0.02)] -= 3
    x[(temp == 24) & (map >= 0.02) & (map < 0.04)] -= 2
    x[(temp == 24) & (map >= 0.04) & (map < 0.08)] -= 1
    x[(temp == 24) & (map >= 0.93) & (map < 0.96)] += 1
    x[(temp == 24) & (map >= 0.96) & (map < 0.98)] += 2
    x[(temp == 24) & (map >= 0.98) & (map < 0.99)] += 3
    x[(temp == 24) & (map >= 0.99)] += 4

    x[(temp == 25) & (map < 0.01)] -= 4
    x[(temp == 25) & (map >= 0.01) & (map < 0.02)] -= 3
    x[(temp == 25) & (map >= 0.02) & (map < 0.04)] -= 2
    x[(temp == 25) & (map >= 0.04) & (map < 0.07)] -= 1
    x[(temp == 25) & (map >= 0.94) & (map < 0.96)] += 1
    x[(temp == 25) & (map >= 0.96) & (map < 0.98)] += 2
    x[(temp == 25) & (map >= 0.98) & (map < 0.99)] += 3
    x[(temp == 25) & (map >= 0.99)] += 4

    x[(temp == 26) & (map < 0.02)] -= 3
    x[(temp == 26) & (map >= 0.02) & (map < 0.04)] -= 2
    x[(temp == 26) & (map >= 0.04) & (map < 0.06)] -= 1
    x[(temp == 26) & (map >= 0.94) & (map < 0.97)] += 1
    x[(temp == 26) & (map >= 0.97) & (map < 0.99)] += 2
    x[(temp == 26) & (map >= 0.99)] += 3

    x[(temp == 27) & (map < 0.01)] -= 3
    x[(temp == 27) & (map >= 0.01) & (map < 0.03)] -= 2
    x[(temp == 27) & (map >= 0.03) & (map < 0.06)] -= 1
    x[(temp == 27) & (map >= 0.95) & (map < 0.97)] += 1
    x[(temp == 27) & (map >= 0.97) & (map < 0.99)] += 2
    x[(temp == 27) & (map >= 0.99)] += 3

    x[(temp == 28) & (map < 0.01)] -= 3
    x[(temp == 28) & (map >= 0.01) & (map < 0.03)] -= 2
    x[(temp == 28) & (map >= 0.03) & (map < 0.05)] -= 1
    x[(temp == 28) & (map >= 0.96) & (map < 0.98)] += 1
    x[(temp == 28) & (map >= 0.98) & (map < 0.99)] += 2
    x[(temp == 28) & (map >= 0.98)] += 3

    x[(temp == 29) & (map < 0.01)] -= 3
    x[(temp == 29) & (map >= 0.01) & (map < 0.02)] -= 2
    x[(temp == 29) & (map >= 0.02) & (map < 0.04)] -= 1
    x[(temp == 29) & (map >= 0.96) & (map < 0.98)] += 1
    x[(temp == 29) & (map >= 0.98)] += 2

    x[(temp == 30) & (map < 0.02)] -= 2
    x[(temp == 30) & (map >= 0.02) & (map < 0.04)] -= 1
    x[(temp == 30) & (map >= 0.96)] += 1

    x[(temp == 31) & (map < 0.02)] -= 1

    return x

def input4_qsa(x, device):
    '''
    x = torch.floor( x / 5 )
    output = x.clamp_max(31) * 5 + 2
    
    # non-ideal
    B, OHOW, K = x.shape
    non_ideal_ratio = torch.tensor([2, 10, 22, 33, 84, 2898, 84, 37, 17, 10, 3], dtype=torch.float)
    #non_ideal_ratio = torch.tensor([0, 0, 0, 0, 0, 3200, 0, 0, 0, 0, 0], dtype=torch.float)
    non_ideal_effect = (torch.multinomial(non_ideal_ratio, B*OHOW*K, replacement=True) - 5).view(B, OHOW, K).to(device)
    output += non_ideal_effect
    '''
    # way-2
    x = torch.floor(x / 5).clamp_max(31)
    output = w_nonideal(x) * 5 + 2

    return output

def input2_qsa(x, device):
    '''
    output = x.clamp_max(31)
    
    # non-ideal
    B, OHOW, K = x.shape
    non_ideal_ratio = torch.tensor([0, 3, 17, 32, 64, 2980, 55, 29, 16, 4, 0], dtype=torch.float)
    #non_ideal_ratio = torch.tensor([0, 0, 0, 0, 0, 3200, 0, 0, 0, 0, 0], dtype=torch.float)
    non_ideal_effect = (torch.multinomial(non_ideal_ratio, B*OHOW*K, replacement=True) - 5).view(B, OHOW, K).to(device)
    output += non_ideal_effect
    '''
    # way-2
    x = x.clamp_max(31)
    output = wo_nonideal(x)
    
    return output

def writeout_dis(name, convX, combX_dis):
    for i in list(convX.reshape(-1)):
        combX_dis[int(i)] += 1
    
    with open('./mac_distribution/' + name + '.dat', 'w') as f:
        for data in combX_dis:
            f.write(str(data.item())+'\n')
        f.close()
    return 0

def CIM_MAC(x_int, w_int, comb_list, device):
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
    '''
    conv5_msb = msb_x.matmul(w5)
    conv5_lsb = lsb_x.matmul(w5)
    conv4_msb = msb_x.matmul(w4)
    conv4_lsb = lsb_x.matmul(w4)
    conv3_msb = msb_x.matmul(w3)
    conv3_lsb = lsb_x.matmul(w3)
    conv2_msb = msb_x.matmul(w2)
    conv2_lsb = lsb_x.matmul(w2)
    conv1_msb = msb_x.matmul(w1)
    conv1_lsb = lsb_x.matmul(w1)
    conv0_msb = msb_x.matmul(w0)
    conv0_lsb = lsb_x.matmul(w0)
    '''
    conv5 = x_float.matmul(w5)
    conv4 = x_float.matmul(w4)
    conv3 = x_float.matmul(w3)
    conv2 = x_float.matmul(w2)
    conv1 = x_float.matmul(w1)
    conv0 = x_float.matmul(w0)
    
    # writeout_dis('conv5', conv5, comb_list[5])
    # writeout_dis('conv4', conv4, comb_list[4])
    # writeout_dis('conv3', conv3, comb_list[3])
    # writeout_dis('conv2', conv2, comb_list[2])
    # writeout_dis('conv1', conv1, comb_list[1])
    # writeout_dis('conv0', conv0, comb_list[0])
    
    # Variation
    conv7_msb = input2_qsa(conv7_msb, device)
    conv7_lsb = input2_qsa(conv7_lsb, device)
    conv6_msb = input2_qsa(conv6_msb, device)
    conv6_lsb = input2_qsa(conv6_lsb, device)
    '''
    conv5_msb = input2_qsa(conv5_msb, device)
    conv5_lsb = input2_qsa(conv5_lsb, device)
    conv4_msb = input2_qsa(conv4_msb, device)
    conv4_lsb = input2_qsa(conv4_lsb, device)
    conv3_msb = input2_qsa(conv3_msb, device)
    conv3_lsb = input2_qsa(conv3_lsb, device)
    conv2_msb = input2_qsa(conv2_msb, device)
    conv2_lsb = input2_qsa(conv2_lsb, device)
    conv1_msb = input2_qsa(conv1_msb, device)
    conv1_lsb = input2_qsa(conv1_lsb, device)
    conv0_msb = input2_qsa(conv0_msb, device)
    conv0_lsb = input2_qsa(conv0_lsb, device)
    '''
    
    conv5 = input4_qsa(conv5, device)
    conv4 = input4_qsa(conv4, device)
    conv3 = input4_qsa(conv3, device)
    conv2 = input4_qsa(conv2, device)
    conv1 = input4_qsa(conv1, device)
    conv0 = input4_qsa(conv0, device)

    
    total = ((conv7_msb * 4 + conv7_lsb) * -128 + 
            (conv6_msb * 4 + conv6_lsb) * 64   + 
            (conv5) * 32 + (conv4) * 16 + (conv3) * 8  +
            (conv2) * 4  + (conv1) * 2  + (conv0))


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
                 comb_list=None,):
        super(Conv2d_CIM_SRAM, self).__init__(in_channels, out_channels, kernel_size, stride, padding,
                                              dilation, groups, bias)
        self.bit = bit
        self.pwr_coef =  2**(bit - 1) 
        self.Round_w = RoundFn_LLSQ.apply
        self.Round_b = RoundFn_Bias.apply
        self.bias_flag = bias
        self.comb_list = comb_list
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
            outputs = self.sram_cim_conv(x, w_q, self.stride, self.padding, alpha, self.alpha_w, LLSQ_b, self.comb_list)
            return outputs

    def extra_repr(self):
        s_prefix = super(Conv2d_CIM_SRAM, self).extra_repr()
        if self.alpha_w is None:
            return '{}, fake'.format(s_prefix)
        return '{}'.format(s_prefix)