import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Function, Variable
from torch.nn.parameter import Parameter
from torch.distributions import Bernoulli

from .Conv2d_quan_mobile import RoundFn_LLSQ, RoundFn_Bias, quan_alpha


# +
#torch.set_default_tensor_type('torch.cuda.FloatTensor')
# -

class Linear_Q(nn.Linear):
    def __init__(self, in_features, out_features,  bias=False, bit=32, extern_init=False, init_model=nn.Sequential()):
        super(Linear_Q, self).__init__(
            in_features, out_features,  bias)
        self.bit = bit
        self.pwr_coef = 2 ** (bit - 1)
        #self.alpha_w = Variable(torch.rand(1) ).cuda()
        
        #self.register_parameter('alpha_w', Parameter(torch.rand(1)))
        #self.register_parameter('alpha_bias', Parameter(torch.rand( 1)))
        # if bias:
        #     self.register_buffer('alpha_b', self.alpha_bias)
        #nn.init.kaiming_normal_(self.alpha , mode='fan_out', nonlinearity='relu')
        # self.weights_q = ACT_WQ(bit,out_channels=out_features,quan_flag =0)
        # self.bias_q = ACT_WQ(bit,out_channels=out_features,quan_flag = 0)
        nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')
        if extern_init:
            param=list(init_model.parameters())
            self.weight=Parameter(param[0])
            if bias:
                self.bias=Parameter(param[1])
        self.Round_w = RoundFn_LLSQ.apply
        self.Round_b = RoundFn_Bias.apply
        if bit < 0:
            self.alpha_w = None
            self.init_state = 0
        else:
            self.alpha_w = Parameter(torch.rand( 1))
            self.register_buffer('init_state', torch.zeros(1))
        
    def forward(self, x):
        if self.bit == 32:
            return F.linear(
                x, self.weight, self.bias)
        else:
            if self.training and self.init_state == 0:
                self.alpha_w.data.copy_(self.weight.detach().abs().max() / (self.pwr_coef + 1))
                self.init_state.fill_(1)
            wq = self.Round_w(self.weight, self.alpha_w, self.pwr_coef, self.bit)
            return F.linear(x, wq, self.bias)
    def extra_repr(self):
        s_prefix = super(Linear_Q, self).extra_repr()
        if self.alpha_w is None:
            return '{}, fake'.format(s_prefix)
        return '{}'.format(s_prefix)
