# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 16:02:28 2018

@author: spinbjy
"""

import torch
import torch.nn as nn
from torch.autograd import Function
import torch.nn.functional as F

class Quantizek(Function):
    
    @staticmethod
    def forward(ctx,x,k):
        n = 2**k - 1
        ctx.save_for_backward(x)
        x = torch.round(x * n) / n
        return x
    
    @staticmethod
    def backward(ctx,grad_output):
        x, = ctx.saved_tensors
        grad_input = grad_output.clone()
        return grad_input,None
    
quantizek = Quantizek.apply

def fw(x,k):
    x = torch.tanh(x)
    x = x / torch.max(torch.abs(x)) * 0.5 + 0.5
    return 2 * quantizek(x,k) - 1

class QuantizekLinear(nn.Linear):
    def __init__(self,k,*args,**kwargs):
        super(QuantizekLinear,self).__init__(*args,**kwargs)
        self.weight_k = fw(self.weight,k)
        #self.bias_k = fw(self.bias,k)
    def forward(self,input):
        out = F.linear(input,self.weight_k,self.bias)
        return out

class QuantizekConv2d(nn.Conv2d):
    def __init__(self,k,*args,**kwargs):
        super(QuantizekConv2d,self).__init__(*args,**kwargs)
        self.weight_k = fw(self.weight,k)
        #self.bias_k = fw(self.weight,k)
    def forward(self,input):
        out = F.conv2d(input,self.weight_k,self.bias,self.stride,self.padding,self.dilation,self.groups)
        return out

class LeNet5_Q(nn.Module):
    def __init__(self):
        super(LeNet5_Q,self).__init__()
        self.conv1 = QuantizekConv2d(2,1,6,kernel_size = 5)
        self.bn_conv1 = nn.BatchNorm2d(6)
        self.conv2 = QuantizekConv2d(2,6,16,kernel_size = 3)
        self.bn_conv2 = nn.BatchNorm2d(16)
        self.fc1 = QuantizekLinear(2,400,50)
        self.fc2 = QuantizekLinear(2,50,10)
    def forward(self,x):
        x = self.conv1(x)
        x = F.relu(F.max_pool2d(self.bn_conv1(x),2))
        x = self.conv2(x)
        x = F.relu(F.max_pool2d(self.bn_conv2(x),2))
        x = x.view(-1,400)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

