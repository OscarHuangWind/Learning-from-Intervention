#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 23:15:09 2023

@author: oscar
"""

import os
import sys
import glob
import random
import argparse
import numpy as np
from tqdm import tqdm 
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch.distributions.categorical import Categorical

from driver_model import DriverModel

import warnings
warnings.filterwarnings("ignore")

# GPU or CPU
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print('Use:', device)

# env = 'RampMerge'
env = 'LeftTurn'
sample = 60
if not os.path.exists('./driver_model/{}'.format(env)):
    os.makedirs('./driver_model/{}'.format(env))
    
class DatasetWrapper(Dataset):
    def __init__(self, data1, data2):
        self.data1 = data1
        self.data2 = data2

    def __len__(self):
        return len(self.data1)

    def __getitem__(self, index):
        x1 = self.data1[index]
        x2 = self.data2[index]
        return x1, x2

def train(epoch, model):

    train_loss = 0.0
    total = 0
    correct = 0
    for i, (obs_data, act_data) in enumerate(train_loader):
        observation = obs_data.to(device)
        action = act_data.type(torch.LongTensor).to(device)

        observation = observation.float().permute(0,3,1,2).to(device)

        predicted_distribution = model.forward(observation)

        optimizer.zero_grad()
        
        if loss_type ==  'RKL':
            loss = RKL_loss(F.log_softmax(action.float(), dim=1),
                           F.softmax(predicted_distribution, dim=1))
        else:
            loss = criterion(predicted_distribution, action)
        
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
        optimizer.step()
        train_loss += loss.detach().cpu().numpy().mean()
        
        _, predicted = torch.max(predicted_distribution.data, 1)
        total += action.size(0)
        
        if loss_type == 'RKL':
            _, label = torch.max(action, axis=1)
            correct += (predicted == label).sum().item()
        else:
            correct += (predicted == action).sum().item()

    accuracy = correct / total
    return round(train_loss/(i), 4), round(accuracy * 100, 2)

def val(epoch, model):
    val_loss = 0.0
    total = 0.0
    correct = 0
    model.eval()
    with torch.no_grad():
        for i, (obs_data, act_data) in enumerate(val_loader):
            observation = obs_data.to(device)
            action = act_data.type(torch.LongTensor).to(device)
    
            observation = observation.float().permute(0,3,1,2).to(device)

            predicted_distribution = model.forward(observation)
            _, predicted = torch.max(predicted_distribution.data, 1)
            total += action.size(0)

            if loss_type ==  'RKL':
                loss = RKL_loss(F.log_softmax(action.float(), dim=1),
                           F.softmax(predicted_distribution, dim=1))
                _, label = torch.max(action, axis=1)
                correct += (predicted == label).sum().item()
            else:
                loss = criterion(predicted_distribution, action)
                correct += (predicted == action).sum().item()
    
            val_loss += loss.detach().cpu().numpy().mean()

            # if (i % 10 == 9):
            #     print('Iter:', i, 'Val Loss:', val_loss/i)
    accuracy = correct / total
    return round(val_loss/i, 4), round(accuracy * 100, 2)

if __name__ == "__main__":

    # load and process data
    OBS = []
    ACT = []
    loss_type = 'RKL' #Reverse KL-divergence
    
    path = os.getcwd()
    directory = '/expert_data/'
    
    try:
        #### Fill with the driver name #####
        driver_list = ['oscar_seed6']
        ####################################
        
        if len(driver_list) == 0:
            print(x)
            
        for driver in driver_list:
            files = glob.glob(path + directory + env + '/' + driver + '/*.npz')
            if sample < len(files):
                files = random.sample(files, sample)
            
            for file in files:
                obs = np.load(file)['obs']
                act = np.load(file)['act']
            
                for i in range(obs.shape[0]):
                    OBS.append(obs[i])
                    ACT.append(act[i])
        
        obs_dataset = np.array(OBS, dtype=np.float32)
        act_dataset = np.array(ACT, dtype=np.float32)
        ind = np.where(act_dataset == 99.0)
        obs_dataset = np.delete(obs_dataset, ind, axis=0)
        act_dataset = np.delete(act_dataset, ind, axis=0)
        
        if loss_type == 'RKL':
            one_hot_act = np.zeros((act_dataset.size, int(act_dataset.max()) + 1))
            one_hot_act[np.arange(act_dataset.size), act_dataset.astype(np.uint8)] = 1
            act_dataset = one_hot_act
        
        ######### Trainining ########
        iteration = 300
        lr = 1e-4
        # set up ensemble
        channel = 9
        action_dim = 4
        ensemble = [DriverModel(channel, action_dim, seed=i).to(device) for i in range(1, 21)]
    
        for idx, model in enumerate(ensemble):
            
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            seed = random.randint(1, 1000)
            generator = torch.Generator().manual_seed(seed)
            
            # selected dataset (80%)
            obs_selected_size = int(0.8*len(obs_dataset))
            obs_remained_size = len(obs_dataset) - obs_selected_size
            selected_dataset, _ = random_split(obs_dataset, [obs_selected_size, obs_remained_size], generator=generator)
            selected_idx = selected_dataset.indices
            obs_selected_dataset = obs_dataset[selected_idx]
            act_selected_dataset = act_dataset[selected_idx]
            
            # Train and Validation dataset (0.8:0.2)
            obs_train_size = int(0.8*len(obs_selected_dataset))
            obs_val_size = len(obs_selected_dataset) - obs_train_size
            obs_train_set, obs_val_set = random_split(obs_selected_dataset, [obs_train_size, obs_val_size], generator=generator)
    
            obs_train_idx = obs_train_set.indices
            obs_val_idx = obs_val_set.indices
            
            # sample
            obs_train_sample = obs_selected_dataset[obs_train_idx]
            obs_val_sample = obs_selected_dataset[obs_val_idx]
            act_train_sample = act_selected_dataset[obs_train_idx]
            act_val_sample = act_selected_dataset[obs_val_idx]
            
            train_ensemble = DatasetWrapper(obs_train_sample, act_train_sample)
            val_ensemble = DatasetWrapper(obs_val_sample, act_val_sample)
            
            # train hyperparameters
            batch_size = 64
            num_workers = 4
            
            train_loader = \
                DataLoader(train_ensemble, batch_size=batch_size, shuffle=True, num_workers=num_workers, generator=generator)
            val_loader = \
                DataLoader(val_ensemble, batch_size=batch_size, shuffle=True, num_workers=num_workers, generator=generator)
            
            print('===== Training Ensemble Model {} ====='.format(idx+1))
            optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    
            if loss_type == 'RKL':
                RKL_loss = nn.KLDivLoss()
            else:
                criterion = nn.CrossEntropyLoss()
    
            file_name = "driver_{}_{}".format(loss_type, idx+1)
        
            min_val_loss = 10
            max_val_acc = 0.5
            val_low_idx = 0
            train_loss_list = []
            val_loss_list = []
            train_acc_list = []
            val_acc_list = []
            fig = plt.figure()
            ax = plt.subplot()
                    
            for epoch in tqdm(range(0, iteration), ascii=True):
                train_loss_epoch, acc_train = train(epoch, model)
                val_loss_epoch, acc_val = val(epoch, model)
            
                train_loss_list.append(train_loss_epoch)
                train_acc_list.append(acc_train)
                val_loss_list.append(val_loss_epoch)
                val_acc_list.append(acc_val)
                
                print('Ensemble:%i, Epoch:%i, Train and Validation loss are:%f, %f' % (idx+1, epoch, train_loss_epoch, val_loss_epoch))
                print('Ensemble:%i, Epoch:%i, Train and Validation accuracy are:%f, %f' % (idx+1, epoch, acc_train, acc_val))
    
                if val_acc_list[-1] >= max_val_acc:
                    val_low_idx = epoch
                    max_val_acc = val_acc_list[-1]
                    min_val_loss = val_loss_list[-1]
                    print("Save the model at Episode:%i" %(epoch))
                    torch.save(model.state_dict(), '%s/%s_actor.pth' % ("./driver_model/{}".format(env), file_name))
            
                if (int(epoch) + 1 == iteration):
                    ax.scatter(val_low_idx, min_val_loss, marker='*', s=128, color='cornflowerblue', label='Lowest Validation Loss Epoch')
            
                if (int(epoch) + 1 == iteration):
                    ax.plot(np.arange(len(train_loss_list)), train_loss_list, label='Train Loss', color='lightseagreen')
                    ax.plot(val_loss_list, label='Validation Loss', color='tomato')
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    ax.set_xlabel('Epoch')
                    ax.set_ylabel('RMSE Loss')
                    ax.legend(frameon=False)
                    plt.title('Driver:' + driver_list[-1] + ', Accuracy: ' + str(max_val_acc) + '%')
                    plt.show()
    except:
        print('Please add folder (driver) name at line 140 based on the collected data.')
