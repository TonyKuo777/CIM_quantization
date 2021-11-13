import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



class quantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, k):
        if k == 32:
            return input
        elif k == 1:
            output = torch.sign(input)
        else:
            n = float(2 ** k - 1)
            output = torch.round(input * n ) / n
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input, None



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
            qx = (2.0 * self.quantize(qx, self.bitW) - 1.0)      #* max_x
            
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
    def __init__(self, bitA=32):
        super(Act_Q, self).__init__()
        self.bitA = bitA
        self.quantize = quantize().apply
    
    def forward(self, x):
        if self.bitA==32:
            # max(x, 0.0)
            qa = torch.nn.functional.relu(x)
        else:
            # min(max(x, 0), 1)
            qa = self.quantize(torch.clamp(x, 0.0, 1.0), self.bitA)
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

def FP2INT(x, bits):
    '''
    From floating point to int
    Args:
        x: input data (FP)
        bits: transfer bits
    
    Return:
        intger data
    '''
    return torch.round((x * 2**(bits-1))).int()             #.astype('int32')

def int2bin(number, bits):
    """ Return the 2'complement binary representation of a number """
    bits_1 = bits - 1
    if number < 0:
        bin_num = bin((1 << bits_1) + number)
    else:
        bin_num = bin(number)
    bin_num = bin_num[2:]
    while len(bin_num) < bits:
        bin_num = '0' + bin_num
    return bin_num

def bin_(data, bits):
    #kernel_num, channel = data.shape
    #bin_data = torch.zeros_like(data)
    _0_bin_data = []
    _1_bin_data = []
    _2_bin_data = []
    _3_bin_data = []
    _4_bin_data = []
    _5_bin_data = []
    _6_bin_data = []
    _7_bin_data = []
    #for k in range(kernel_num):
    for _16_c in range(len(data)):
        temp = data[_16_c]
        line = int2bin(temp, bits)
        _0_bin_data.append(int(line[7]))
        _1_bin_data.append(int(line[6]))
        _2_bin_data.append(int(line[5]))
        _3_bin_data.append(int(line[4]))
        _4_bin_data.append(int(line[3]))
        _5_bin_data.append(int(line[2]))
        _6_bin_data.append(int(line[1]))
        _7_bin_data.append(int(line[0]))
    return _0_bin_data, _1_bin_data, _2_bin_data, _3_bin_data, _4_bin_data, _5_bin_data, _6_bin_data, _7_bin_data

def in4_q_sa(_k_conv, power):
    sa_out = _k_conv // 5                   # 0 - 48
    sa_out = torch.clamp(sa_out, max=31)    # 0 - 31
    real_value = sa_out * 5 * 2 ** power    # 0 - 155 * w_weight
    
    return real_value

def in2_q_sa(_k_conv_Xsb, Xsb):
    sa_out = torch.clamp(_k_conv_Xsb, max=31)
    if Xsb == 'msb':
        input_bit_slice = sa_out * 4
    elif Xsb == 'lsb':
        input_bit_slice = sa_out
    else:
        raise NotImplementedError

    return input_bit_slice
    

def conv_sram(inp_group, qw_group):
    qw_group_int = FP2INT(qw_group, 8)
    _0_bin_data, _1_bin_data, _2_bin_data, _3_bin_data, _4_bin_data, _5_bin_data, _6_bin_data, _7_bin_data = bin_(qw_group_int, 8)
    for a in range(len(_0_bin_data)):          
        _0_conv = _0_bin_data[a] * inp_group.sum()
        _1_conv = _1_bin_data[a] * inp_group.sum()
        _2_conv = _2_bin_data[a] * inp_group.sum()
        _3_conv = _3_bin_data[a] * inp_group.sum()
        _4_conv = _4_bin_data[a] * inp_group.sum()
        _5_conv = _5_bin_data[a] * inp_group.sum()
        _6_conv_msb = _6_bin_data[a] * (inp_group // 4).sum()
        _6_conv_lsb = _6_bin_data[a] * (inp_group % 4).sum()
        _7_conv_msb = _7_bin_data[a] * (inp_group // 4).sum()
        _7_conv_lsb = _7_bin_data[a] * (inp_group % 4).sum()

        # bit 0 - 5
        real_value_0 = in4_q_sa(_0_conv, 0)
        real_value_1 = in4_q_sa(_1_conv, 1)
        real_value_2 = in4_q_sa(_2_conv, 2)
        real_value_3 = in4_q_sa(_3_conv, 3)
        real_value_4 = in4_q_sa(_4_conv, 4)
        real_value_5 = in4_q_sa(_5_conv, 5)
        
        # bit 6
        input_bit_slice_6_msb = in2_q_sa(_6_conv_msb, Xsb='msb')
        input_bit_slice_6_lsb = in2_q_sa(_6_conv_lsb, Xsb='lsb')
        real_value_6 = (input_bit_slice_6_msb + input_bit_slice_6_lsb) * 2 ** 6

        # bit 7
        input_bit_slice_7_msb = in2_q_sa(_7_conv_msb, Xsb='msb')
        input_bit_slice_7_lsb = in2_q_sa(_7_conv_lsb, Xsb='lsb')
        real_value_7 = (input_bit_slice_7_msb + input_bit_slice_7_lsb) * (-128)

        total_real_value = real_value_0 + real_value_1 + real_value_2 + real_value_3 + real_value_4 + real_value_5 + real_value_6 + real_value_7

    return total_real_value   

def un_fold(input, q_weight, stride, padding):
    input_slices = input.split(16, 1)                 #class 'tuple'  4 groups         
    qw_slices = q_weight.split(16, 1)                 #class 'tuple'  4 groups
    o_h = (input - q_weight + 2 * padding) / stride + 1
    
    # initialize output_feature_map
    ofm = torch.empty(input.shape[0], q_weight.shape[0], o_h, o_h)
    output_feature_map = torch.zeros_like(ofm)
    
    for input_slice, qw_slice in zip(input_slices, qw_slices):
        k_h = qw_slice.shape[2]                 # kernel height
        k_w = qw_slice.shape[3]                 # kernel width
        inp_unf = torch.nn.functional.unfold(input_slice, kernel_size=(k_h, k_w), stride=stride, padding=padding)
        w = qw_slice.view(qw_slice.size(0), -1).t()
        i = inp_unf.transpose(1, 2)
        out_unf = torch.empty(i.shape[0], i.shape[1], w.shape[1])        # initialize out_unf
        for bs in range(i.shape[0]):     # batch size
            for square in range(i.shape[1]):
                for k_num in range(w.shape[1]):
                    inp_groups = i[bs, square, :].split(16)
                    qw_groups = w[:, k_num].split(16)
                    sliced_output_pixel = 0
                    for inp_group, qw_group in zip(inp_groups, qw_groups):
                        sliced_output_pixel += conv_sram(inp_group, qw_group)    # 144*144
                        
                    out_unf[bs, square, k_num] = sliced_output_pixel
        #print(out_unf.shape)                
        out_unf = out_unf.transpose(1, 2)
        output_feature_map = torch.nn.functional.fold(out_unf, (o_h, o_h), (1, 1)) + output_feature_map
        
    return output_feature_map

# def Conv2d_R2Q(in_channels, out_channels, kernel_size, stride, padding=0, bitW=32, bitO=32, sub_channel='v', mode='ReRam', dilation=1, groups=1, bias=False):
#     q_weight = fw()

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
                
            #groups = self.in_channels // 16     #self.sub_channel
            #self.sub_channel = input[1] // 16
            output_feature_map = un_fold(input, q_weight, stride=self.stride, padding=self.padding)
            # input_slices = input.split(16, 1)                 #class 'tuple'  4 groups         #input.reshape(-1, groups, self.sub_channel, in_H, in_W)
            # qw_slices = q_weight.split(16, 1)                #class 'tuple'  4 groups         #q_weight.reshape(self.out_channels, groups, self)
                        
            
            #for input_slice, qw_slice in zip(input_slices, qw_slices):
                
                
                # for oc in range(self.out_channels):
                #     for b in range(len(input_slice[0])):             # batch size 
                #         for h in range(len(input_slice[2])):         #split to 1 * 1 * 16
                #             for w in range(len(input_slice[3])):
                                # conv_sub_layers = [conv_sram(input_slice[b, :, h, w], qw_slice[oc, :, h, w],
                                # self.bias, self.stride, self.padding, self.dilation, self.groups)]


            # for c, qw_slice in enumerate(qw_slices):
            #     q_slice_int[c] = FP2INT(qw_slice, 8)
            #     _0_bin_data[c], _1_bin_data[c], _2_bin_data[c], _3_bin_data[c], _4_bin_data[c], _5_bin_data[c], _6_bin_data[c], _7_bin_data[c] = bin_(q_slice_int[c], 8)
            #     for a in range(len(_0_bin_data)):
            #         _0_conv = _0_bin_data[a] * input_slice
            #         _1_conv = _1_bin_data[a] * input_slice
            #         _2_conv = _2_bin_data[a] * input_slice
            #         _3_conv = _3_bin_data[a] * input_slice
            #         _4_conv = _4_bin_data[a] * input_slice
            #         _5_conv = _5_bin_data[a] * input_slice
            #         _6_conv = _6_bin_data[a] * input_slice
            #         _7_conv = _7_bin_data[a] * input_slice
                    

            #     input_slice
            #     conv_sub_layers = [F.conv2d(input_slice, _0_bin_data, self.bias, self.stride, self.padding, self.dilation, self.groups)]

            # SA_sub_out = self.fo(conv_sub_layers)
            # outputs = SA_sub_out.sum(4)
            # return outputs
            return output_feature_map
