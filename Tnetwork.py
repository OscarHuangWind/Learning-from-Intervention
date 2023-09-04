#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 14:03:18 2023

@author: oscar
"""

import torch
import torch.nn as nn
import numpy as np
import random

from vit_backbone import SimpleViT

if torch.cuda.is_available():
    device = torch.device("cuda", 0 if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")
print('Use:', device)

##############################################################################
class Net(nn.Module):

    def __init__(self, height, width, channel, num_outputs, dueling, preference, seed):
        super(Net, self).__init__()
        self.dueling = dueling
        self.preference = preference
        self.height = height
        self.width = width
        self.linear_dim = 128
        self.hidden_dim = 512
        self.feature_dim = 256
        
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        self.trans = SimpleViT(
            image_size = (self.height, self.width ),
            patch_size = (int(self.height/5), int(self.width/5)),
            num_classes = 2,
            dim = self.feature_dim,
            depth = 2,
            heads = 8,
            mlp_dim = self.hidden_dim,
            channels = channel
        )
        
        if (self.dueling):
            self.advantage_func = nn.Sequential(
                nn.Linear(self.feature_dim, self.linear_dim),
                nn.LayerNorm(self.linear_dim),
                nn.ReLU(),
                nn.Linear(self.linear_dim, num_outputs)
                )
            
            self.state_value_func = nn.Sequential(
                nn.Linear(self.feature_dim, self.linear_dim ),
                nn.LayerNorm(self.linear_dim),
                nn.ReLU(),
                nn.Linear(self.linear_dim , 1)
                )

        elif (self.preference):
            self.preference_head = nn.Sequential(
                nn.Linear(self.feature_dim, self.linear_dim),
                nn.LayerNorm(self.linear_dim),
                nn.ReLU(),
                nn.Linear(self.linear_dim, num_outputs),
                )

            self.q_head = nn.Sequential(
                nn.Linear(self.feature_dim, self.linear_dim),
                nn.LayerNorm(self.linear_dim),
                nn.ReLU(),
                nn.Linear(self.linear_dim, num_outputs),
                )
        else:
            self.fc = nn.Sequential(
                nn.Linear(self.feature_dim, self.linear_dim),
                nn.LayerNorm(self.linear_dim),
                nn.ReLU(),
                nn.Linear(self.linear_dim, num_outputs)
                )

    def forward(self, x):
        x = x.to(device)
        x = self.trans(x)
        x = x.contiguous().view(-1, self.feature_dim)
        if (self.dueling):
            advantage_vec = self.advantage_func(x)
            value_scalar = self.state_value_func(x)
            x = value_scalar + advantage_vec - advantage_vec.mean()
            return x
        elif (self.preference):
            q_value = self.q_head(x)
            action_distribution = self.preference_head(x)
            normalize = nn.Softmax(dim=1)
            action_distribution = normalize(action_distribution)
            return action_distribution, q_value
        else:
            x = self.fc(x)
            return x
        
##############################################################################
