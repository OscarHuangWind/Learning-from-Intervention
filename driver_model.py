#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 14:28:41 2023

@author: oscar
"""

import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

# import warnings

# GPU or CPU
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print('Use:', device)

# define imitation learning actor
class DriverModel(nn.Module):
    def __init__(self, channel, action_dim, seed):
        super(DriverModel, self).__init__()

        self.action_dim = action_dim
        self.width = 120
        self.height = 120
        self.kernel_1st = 3
        self.stride_1st = 2
        self.kernel_2nd = 3
        self.stride_2nd = 2
        self.kernel_3rd = 3
        self.stride_3rd = 2
        self.feature_dim = 128
        self.hidden_dim = 64 

        self.linear_input_size = self.linear_size_input()
    
        self.feature_net = nn.Sequential(
            nn.Conv2d(channel, self.hidden_dim, kernel_size=self.kernel_1st, stride=self.stride_1st),
            nn.ReLU(),
            nn.BatchNorm2d(self.hidden_dim),
            nn.Conv2d(self.hidden_dim, self.feature_dim, kernel_size=self.kernel_2nd, stride=self.stride_2nd),
            nn.ReLU(),
            nn.BatchNorm2d(self.feature_dim),
        )
        
        self.mlp = nn.Sequential(
            nn.Dropout2d(p=0.5),
            nn.Linear(self.linear_input_size, action_dim),
            )

        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        x = x.to(device)
        x = self.feature_net(x)
        x = x.contiguous().view(-1, self.linear_size_input())

        x = self.mlp(x)
        distribution = self.softmax(x)
        return distribution
    
    def select_action(self, x):
        with torch.no_grad():
            x = torch.FloatTensor(x.transpose(2,0,1)[None].copy()) / 255.0
            distribution = self.forward(x)
            m = Categorical(distribution)
            action = m.sample()
            return distribution, action
    
    def linear_size_input(self):
        convw = self.conv2d_size_2nd(self.conv2d_size_1st(self.width))
        convh = self.conv2d_size_2nd(self.conv2d_size_1st(self.height))
        return convw * convh * self.feature_dim

    def conv2d_size_1st(self, size):
        return (size - (self.kernel_1st - 1) - 1) // self.stride_1st + 1
    
    def conv2d_size_2nd(self, size):
        return (size - (self.kernel_2nd - 1) - 1) // self.stride_2nd  + 1
   
    def conv2d_size_3rd(self, size):
        return (size - (self.kernel_3rd - 1) - 1) // self.stride_3rd  + 1

    def conv2d_size_4th(self, size, kernel_size = 2, stride = 2):
        return (size - (kernel_size - 1) - 1) // stride  + 1
