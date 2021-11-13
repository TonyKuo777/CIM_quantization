import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.modules.utils import _pair
from torch.nn.parameter import Parameter
from torch.nn import init
import math

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
        return x

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
    def forward(ctx, input, weight, stride, padding, bias=None):
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
                x_int = torch.round(input_unfold[:, :, i, :] * 255).int()   # 255 <- 2**8 - 1
                w_int = torch.round(torch.clamp(weight_unfold[i, :, :] * 128, max=127)).int()   # 128 <- 2**(8-1)
                
                msb_x_int = x_int >> 4
                lsb_x_int = x_int & 15
        
                output_unfold += ( (CIM_MAC(msb_x_int, w_int) << 4) + CIM_MAC(lsb_x_int, w_int) )
                
            output_unfold = output_unfold.transpose(1, 2)                            # [B, K, OH*OW]
            output += torch.nn.functional.fold(output_unfold, (OH, OW), (1, 1))
        output = output / (255*127)   # 8a8w

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
    
        return grad_input, grad_weight, None, None, None

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

def input4_qsa(x):
    x = torch.floor(x / 5)
    return x.clamp_max(31) * 5
    
def input2_qsa(x):
    return x.clamp_max(31)

def CIM_MAC(x_int, w_int):
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
    conv5 = x_float.matmul(w5)
    conv4 = x_float.matmul(w4)
    conv3 = x_float.matmul(w3)
    conv2 = x_float.matmul(w2)
    conv1 = x_float.matmul(w1)
    conv0 = x_float.matmul(w0)
    
    
    # Varivation
    conv7_msb = input2_qsa(conv7_msb)
    conv7_lsb = input2_qsa(conv7_lsb)
    conv6_msb = input2_qsa(conv6_msb)
    conv6_lsb = input2_qsa(conv6_lsb)
    conv5 = input4_qsa(conv5)
    conv4 = input4_qsa(conv4)
    conv3 = input4_qsa(conv3)
    conv2 = input4_qsa(conv2)
    conv1 = input4_qsa(conv1)
    conv0 = input4_qsa(conv0)
    
    
    total = ((conv7_msb * 4 + conv7_lsb) * -128 + 
            (conv6_msb * 4 + conv6_lsb) * 64   + 
            conv5 * 32 + conv4 * 16 + conv3 * 8  +
            conv2 * 4  + conv1 * 2  + conv0)

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
        
        self.sub_flag = sub_channel == 'v' or self.in_channels==1 or self.in_channels==3
        self.fw = fw(bitW)
        self.macs = self.weight.shape[1:].numel()
        self.fo = fo(bitO)
        self.sram_cim_conv = sram_cim_conv.apply
        
    def forward(self, input, order=None):
        q_weight = self.fw(self.weight)
 
        #conv_sub_layers = F.conv2d(input, q_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        conv_sub_layers_sram = self.sram_cim_conv(input, q_weight, self.stride[0], self.padding[0])
        
        outputs = self.fo(conv_sub_layers_sram)
        return outputs