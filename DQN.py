#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 17:08:43 2023

@author: oscar
"""

import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch.nn.utils import clip_grad_value_
from torch.distributions.categorical import Categorical

from Tnetwork import Net, device
from cpprb import PrioritizedReplayBuffer

class DQN():
    
    def __init__(self,
                 height,
                 width,
                 channel,
                 pstate_dim,
                 n_obs,
                 n_actions,
                 DOUBLE,
                 DUELING,
                 IMPORTANTSAMPLING,
                 PREFERENCE,
                 ENTROPY,
                 TOGETHER,
                 BATCH_SIZE,
                 GAMMA,
                 EPS_START,
                 EPS_END,
                 EPS_EPOC,
                 THRESHOLD,
                 MEMORY_CAPACITY,
                 seed,
                 ):
        self.height = height
        self.width = width
        self.channel = channel
        self.n_obs = n_obs
        self.n_actions = n_actions
        self.batch_size = BATCH_SIZE
        self.gamma = GAMMA
        self.eps_start = EPS_START
        self.eps_end = EPS_END
        self.eps_epoc = EPS_EPOC
        self.threshold = THRESHOLD
        self.memory_capacity = MEMORY_CAPACITY
        self.seed = seed
        self.double = DOUBLE
        self.dueling = DUELING
        self.preference = PREFERENCE
        self.auto_entropy = ENTROPY
        self.imsamp = IMPORTANTSAMPLING
        self.together = TOGETHER
        
        ##### Hyper Parameters ####
        self.lr = 0.00025
        self.lr_p = 0.0001
        self.lr_temp = 0.001
        self.alpha = 0.95
        self.eps = 0.01
        self.tau = 0.005
        self.steps_done = 0
        self.loss_critic = 0.0
        self.loss_actor = 0.0
        self.loss_entropy = 0.0
        self.loss_engage_q = 0.0
        self.loss_engage_preference = 0.0
        self.eps_threshold = 0.0
        self.q = 0.0
        self.action_distribution = 0.0
        self.target_entropy_ratio = 0.3
        self.temperature_copy = 0.0
        self.default_weight = np.exp(1.0)
        self.policy_guidance = False
        self.value_guidance = False
        self.adaptive_weight = False
        self.ac_dis_policy = np.zeros(self.n_actions)
        self.ac_dis_target = np.zeros(self.n_actions)
        self.q_policy = np.zeros(self.n_actions)
        self.q_target = np.zeros(self.n_actions)

        ##### Fix Seed ####
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        ##### Initializing Net ####
        self.policy_net = Net(height, width, channel, n_actions, DUELING, PREFERENCE, seed)#.to(device)
        self.target_net = Net(height, width, channel, n_actions, DUELING, PREFERENCE, seed)#.to(device)

        if torch.cuda.device_count() > 8:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            self.policy_net = nn.DataParallel(self.policy_net)
            self.target_net = nn.DataParallel(self.target_net)
            self.policy_net.to(device)
            self.target_net.to(device)
        else:
            self.policy_net.to(device)
            self.target_net.to(device)

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        ##### Loss and Optimizer ####
        self.GL = nn.GaussianNLLLoss(reduction='none')
        self.KL = nn.KLDivLoss()
        self.CE = nn.CrossEntropyLoss()
        self.SL = nn.SmoothL1Loss()
        self.normalize = nn.Softmax(dim=1)
        
        self.optimizer = optim.RMSprop(self.policy_net.parameters(),lr=self.lr,
                                       alpha=self.alpha, eps=self.eps)
        self.optimizer_p = optim.RMSprop(self.policy_net.parameters(),lr=self.
                                         lr_p, alpha=self.alpha, eps=self.eps)

        self.scheduler = lr_scheduler.ExponentialLR(self.optimizer, gamma=1.0)
        self.scheduler_p = lr_scheduler.ExponentialLR(self.optimizer_p, gamma=1.0)

        ##### Temperature Adjustment ####
        if self.auto_entropy:
            self.target_entropy = \
                -np.log(1.0 / self.n_actions) * self.target_entropy_ratio
            self.log_temp = torch.zeros(1, requires_grad=True, device=device)
            self.temperature = self.log_temp.exp()
            self.temp_optim = optim.Adam([self.log_temp], lr=self.lr_temp)
        else:
            self.temperature = 0.2 #1.0 / max(1, self.n_actions)

        ##### Replay Buffer ####
        self.replay_buffer = PrioritizedReplayBuffer(self.memory_capacity,
                                          {"obs": {"shape": (self.height,self.width,9),"dtype": np.uint8},
                                           "act": {},
                                           "rew": {},
                                           "next_obs": {"shape": (self.height,self.width,9),"dtype": np.uint8},
                                           "engage": {},
                                           "done": {}},
                                          next_of=("obs"))

    def select_action_deterministic(self, x):
        sample = random.random()
        eps_threshold = 0.0
        with torch.no_grad():
            x = torch.FloatTensor(x.transpose(2,0,1)[None]) / 255.0
            self.steps_done += 1
            if sample > eps_threshold:
                
                if (self.preference):
                    action_distribution, q = self.policy_net.forward(x)
                    action_idx = q.max(1)[1].view(1, 1)
                    self.action_distribution = action_distribution.squeeze(0).cpu().numpy()
                    self.q = q.squeeze(0).cpu().numpy()
                    
                    return self.action_distribution, action_idx
                
                return 1.0, self.policy_net.forward(x).max(1)[1].view(1, 1)
            
            else:
                return torch.tensor([[random.randrange(self.n_actions)]], 
                                    device=device, dtype=torch.long) 

    def select_action(self, x, i_epoc):
        sample = random.random()
        
        self.eps_threshold = self.calc_eps_threshold(i_epoc)
        with torch.no_grad():
            x = torch.FloatTensor(x.transpose(2,0,1)[None].copy()) / 255.0
        
            ##### Greedy Action ####
            if sample > self.eps_threshold:
                
                if (self.preference):
                    action_distribution, q = self.policy_net.forward(x)
                    action_idx = q.max(1)[1].view(1, 1)
                    self.action_distribution = action_distribution.squeeze(0).cpu().numpy()
                    self.q = q.squeeze(0).cpu().numpy()
                    return self.action_distribution, action_idx
                
                q = self.policy_net.forward(x)
                self.q = q.squeeze(0).cpu().numpy()
                q_distribution = self.normalize(q)
                return q_distribution.squeeze(0).cpu().numpy(), q.max(1)[1].view(1, 1)
            
            ##### Stochastic Action ####
            else:
                if (self.preference):
                    
                    action_distribution, q = self.policy_net.forward(x)
                    self.q = q.squeeze(0).cpu().numpy()
                    action_distribution = action_distribution.squeeze(0).cpu().numpy()
                    distribution = action_distribution / action_distribution.sum().tolist()
                    distribution = np.nan_to_num(distribution)
                    
                    return action_distribution, \
                           torch.tensor([[np.random.choice(np.arange(0, self.n_actions),\
                                        p=distribution)]], device=device, dtype=torch.long)

                q = self.policy_net.forward(x)
                self.q = q.squeeze(0).cpu().numpy()
                q_distribution = self.normalize(q)
                return q_distribution.squeeze(0).cpu().numpy(), \
                       torch.tensor([[random.randrange(self.n_actions)]], device=device, dtype=torch.long)

    def calc_eps_threshold(self, i_epoc):
        if (i_epoc <= self.threshold):
            return self.eps_start
        else:
            fraction = min((i_epoc - self.threshold) / self.eps_epoc, 1.0)
            return self.eps_start + fraction * (self.eps_end - self.eps_start)

    def optimize_Q_network(self, state, action, reward, next_state, driver_model,
                       variance_list):

        ##### Sample Batch #####
        data = self.replay_buffer.sample(self.batch_size)
        istates, actions, engages = data['obs'], data['act'], data['engage']
        rewards, next_istates, dones = data['rew'], data['next_obs'], data['done']

        state_batch = torch.FloatTensor(istates).permute(0,3,1,2).to(device) / 255.0
        action_batch = torch.FloatTensor(actions).to(device)
        reward_batch = torch.FloatTensor(rewards).to(device)
        engages = torch.FloatTensor(engages).to(device)
        next_state_batch = torch.FloatTensor(next_istates).permute(0,3,1,2).to(device) / 255.0
        dones = torch.FloatTensor(dones).to(device)

        ##### Q value #####
        if (not self.preference):
            q_policy = self.policy_net.forward(state_batch)
            q_policy_selected = q_policy.gather(1, action_batch.type(torch.int64))
        else:
            action_distribution, q_policy = self.policy_net.forward(state_batch)
            q_policy_selected = q_policy.gather(1, action_batch.long())

        next_q_target = torch.zeros(self.batch_size, device=device)

        if (self.double):
            if (self.preference):
                _, next_q_policy_temp = self.policy_net.forward(next_state_batch)
                _, next_q_target_temp = self.target_net.forward(next_state_batch)
                next_q_target_temp = next_q_target_temp.detach()
            else:
                next_q_policy_temp = self.policy_net.forward(next_state_batch)
                next_q_target_temp = self.target_net.forward(next_state_batch).detach()
                
            max_action_indices = torch.argmax(next_q_policy_temp, dim=1)
            indices_batch = torch.LongTensor(np.arange(self.batch_size))
            next_q_target = next_q_target_temp[indices_batch, max_action_indices]

        elif (self.preference):
            _, next_q_target = self.target_net.forward(next_state_batch)
            next_q_target = next_q_target.max(1)[0].detach()
        else:
            next_q_target = self.target_net.forward(next_state_batch).max(1)[0].detach()

        ##### Q target ####
        q_target = (next_q_target.unsqueeze(1) * self.gamma) + reward_batch

        ##### critic loss ######
        loss_critic = self.SL(q_policy_selected, q_target)
        if torch.isnan(loss_critic):
            print('q loss is nan.')
        self.loss_critic = loss_critic.detach().cpu().numpy()

        ##### engage loss ######
        
        ##### UnHiL, HIRL, EIL ######
        if self.value_guidance:
            engage_index = (engages == 1).nonzero(as_tuple=True)[0]
            if engage_index.numel() > 0:
                states_expert = state_batch[engage_index]
                actions_expert = action_batch[engage_index]
                actions_rl = q_policy[engage_index]
                one_hot_expert_actions = torch.squeeze(F.one_hot(actions_expert.long(),
                                                                 self.n_actions), axis=1)
                expected_var = torch.ones_like(actions_rl)
                
                driver_mean, driver_variance, probability, driver_a =\
                    self.driver_decision(states_expert, driver_model, online=False)
                    
                var_max = max(variance_list)
                var_min = min(variance_list)
                var_selected = driver_variance[np.arange(actions_expert.size(dim=0)),
                                               torch.squeeze(actions_expert).long()]
                x = (var_selected - var_min) / (var_max - var_min + 1e-7)
                engage_weight  = torch.exp(-2 * x + 1)
                
                #####!!!!!! Adaptive Confidence Adjustment !!!!!!#####
                if self.adaptive_weight:
                    loss_engage = (self.GL(actions_rl, one_hot_expert_actions,
                                            expected_var).mean(dim=1) * engage_weight).mean()
                else:
                    loss_engage = (self.GL(actions_rl, one_hot_expert_actions,
                                            expected_var).mean(dim=1)).mean() * self.default_weight

                self.loss_engage_q = loss_engage.detach().cpu().numpy()
            else:
                loss_engage = 0.0
                self.loss_engage_q = loss_engage
                
        ###### IARL, DRL ######
        else:
            loss_engage = 0.0
            self.loss_engage_q = loss_engage

        ##### Overall Loss #####
        self.optimizer.zero_grad()
        if (self.preference and self.together):
            loss_policy = self.policy_gradient(state, action, reward, next_state)
            loss_total = loss_critic + loss_policy + loss_engage
        elif (not self.preference and self.together):
            print('preference is false, but together is true.')
            loss_total = loss_critic + loss_engage
        else:
            loss_total = loss_critic + loss_engage

        ##### Optimization #####
        loss_total.backward()
        clip_grad_value_(self.policy_net.parameters(), 1)
        self.optimizer.step()

        ##### Soft Update Target Network #####
        self.soft_update(self.target_net, self.policy_net, self.tau)

    def policy_gradient(self, state, action, reward, next_state,
                        engage, driver_mean, driver_variance, variance_list):
        
        state_tensor = torch.FloatTensor(state.transpose(2,0,1)[None].copy()).to(device) / 255.0
        next_state_tensor = torch.FloatTensor(state.transpose(2,0,1)[None].copy()).to(device) / 255.0
        action_distribution_policy, q_policy = self.policy_net.forward(state_tensor)
        action_distribution_target, _ = self.target_net.forward(state_tensor)
        _, next_q_target_temp = self.target_net.forward(next_state_tensor)
        q_target = torch.from_numpy(np.array(reward)).to(device) +\
                   next_q_target_temp * self.gamma

        ##### For main function #####
        self.ac_dis_policy = action_distribution_policy.squeeze(0).cpu().detach().numpy()
        self.ac_dis_target = action_distribution_target.squeeze(0).cpu().detach().numpy()
        self.q_policy = q_policy.squeeze(0).cpu().detach().numpy()
        self.q_target = q_target.squeeze(0).cpu().detach().numpy()

        action_distribution_policy = action_distribution_policy.squeeze(0)
        action_distribution_target = action_distribution_target.squeeze(0)
        action_prob_policy = Categorical(action_distribution_policy)

        q_policy = q_policy.squeeze(0)
        q_target = q_target.squeeze(0)

        state_value = torch.matmul(action_distribution_target, q_target)
        advantage_function = (q_target - state_value).detach()

        ###### Loss Function ######
        loss_policy = - torch.matmul(action_prob_policy.probs, advantage_function)
        if torch.isnan(loss_policy):
            print('policy loss is nan.')
        self.loss_policy = loss_policy.detach().cpu().numpy()

        loss_entropy =  - action_prob_policy.entropy().mean()
        if torch.isnan(loss_entropy):
            print('entropy loss is nan.')
        self.loss_entropy = loss_entropy.detach().cpu().numpy()

        if self.policy_guidance and engage:
            loss_engage = self.KL(driver_mean.log(), action_distribution_policy)
            self.loss_engage_preference = loss_engage.detach().cpu().numpy()
            var_max = max(variance_list)
            var_min = min(variance_list)
            prob, ind = torch.max(driver_mean, axis=0)
            var_selected = driver_variance[ind]
            x = (var_selected - var_min) / (var_max - var_min + 1e-7)
            engage_weight  = torch.exp(-2 * x + 1)
            
            #####!!!!!! Adaptive Confidence Adjustment !!!!!!#####
            if self.adaptive_weight:
                loss_policy = loss_policy + loss_engage * engage_weight
            else:
                loss_policy = loss_policy + loss_engage * self.default_weight
            
        elif (self.temperature > 0):
            loss_policy = loss_policy + loss_entropy * self.temperature
        else:
            loss_policy = loss_policy

        ##### Temperature Adjustment ######
        if self.auto_entropy:
            self.optimize_entropy_parameter(loss_entropy.detach())
            self.temperature_copy = self.temperature.detach().squeeze().cpu().numpy()
        else:
            self.temperature_copy = self.temperature

        return loss_policy
        
    def optimize_preference_network(self, state, action, reward, next_state, engage,
                               driver_mean, driver_variance, variance_list):

        ##### Something Wrong #####
        if (not self.preference or self.together):
            print(self.preference, '|', self.together, '|RETURN!')
            return

        loss_policy = self.policy_gradient(state, action, reward, next_state,
                                           engage, driver_mean, driver_variance,
                                           variance_list)
        
        ##### Optimization #####
        self.optimizer_p.zero_grad()
        loss_policy.backward()
        clip_grad_value_(self.policy_net.parameters(), 1)
        self.optimizer_p.step()

    def store_transition(self, s, a, r, s_, engage, d=0):
        self.replay_buffer.add(obs=s,
                act=a,
                rew=r,
                next_obs=s_,
                engage = engage,
                done=d)

    def optimize_entropy_parameter(self, entropy):
        temp_loss = -torch.mean(self.log_temp * (self.target_entropy + entropy))
        self.temp_optim.zero_grad()
        temp_loss.backward()
        self.temp_optim.step()
        self.temperature = self.log_temp.detach().exp()
        
    def soft_update(self, target, source, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - tau) + param.data * tau
            )
    
    def driver_decision(self, observation, driver_ensemble, online):
        policies = []
        variances = []
        
        for driver in driver_ensemble:
            if online:
                policy_distribution, _ = driver.select_action(observation)
            else:
                policy_distribution = driver.forward(observation)
            policies.append(policy_distribution)
        
        policy_matrix = torch.stack(policies)

        ##### Online Human Intervention for DRL #####
        if online:
            policy_matrix = torch.squeeze(policy_matrix)

        mean_policy = torch.mean(policy_matrix, axis=0)
        variance_policy = torch.var(policy_matrix, axis=0)
        
        if online:
            prob, action = torch.max(mean_policy, axis=0)
            # m = Categorical(mean_policy)
            # action = m.sample()
            # prob = mean_policy[action]

        ##### For batch optimization #####
        else:
            prob, action = torch.max(mean_policy, axis=1)

        return mean_policy, variance_policy, prob, action