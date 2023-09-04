#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 17:06:47 2023

@author: oscar
"""
import os
import gym
import sys
import yaml
import torch
import warnings
import statistics
import scipy.stats
import numpy as np
from tqdm import tqdm 
from itertools import count
from collections import deque
import matplotlib.pyplot as plt


sys.path.append('/home/oscar/Dropbox/SMARTS')
from smarts.core.agent import AgentSpec
from smarts.env.hiway_env import HiWayEnv
from smarts.core.controllers import ActionSpaceType
from smarts.core.agent_interface import AgentInterface
from smarts.core.agent_interface import NeighborhoodVehicles, RGB, OGM, DrivableAreaGridMap

from DQN import DQN
from Tnetwork import device
from driver_model import DriverModel

def plot_animation_figure(epoc):
    plt.figure()
    plt.clf()

    plt.subplot(2, 1, 1)
    plt.title(env_name + ' ' + name + ' Double: ' + str(agent.double) +
              ' Save Epoc:' + str(epoc))
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.plot(reward_list)
    plt.plot(reward_mean_list)

    plt.subplot(2, 1, 2)
    plt.xlabel('Episode')
    plt.ylabel('Guidance Rate')
    plt.plot(train_durations, guidance_list)

    plt.pause(0.001)
    plt.tight_layout()
    plt.show()

def smooth(train_duration_list, reward_list, loss_list, smooth_horizon):
    mean1 = np.mean(train_duration_list[-min(len(train_duration_list), smooth_horizon):])
    train_durations_mean_list.append(mean1)
    mean2 = np.mean(loss_list[-min(len(loss_list), smooth_horizon):])
    mean3 = np.mean(reward_list[-min(len(reward_list), smooth_horizon):])
    reward_mean_list.append(mean3)

def preprocess(s, a, r, s_):
    state = s
    next_state  = s_
    action = a.cpu().numpy().squeeze().astype(np.int32)
    reward = np.float32(r)
    
    return state, action, reward, next_state

def evaluate(network, eval_episodes=10, epoch=0):
    ep = 0
    success = int(0)
    avg_reward_list = []
    cumulate_flag = True
    while ep < eval_episodes:
        obs = env.reset()
        s = observation_adapter(obs[AGENT_ID])
        done = False
        reward_total = 0.0 
        frame_skip = 3

        for t in count():
            
            if t > MAX_NUM_STEPS:
                print('Max Steps Done.')
                break
       
            if t < frame_skip:
                ##### Select and perform an action #####
                if mode == 'driver': 
                    _, a = agent.select_action(s)
                else:
                    _, a = agent.select_action_deterministic(s)
                
                action = {AGENT_ID:action_adapter(a)}
                next_state, reward, done, info = env.step(action)
                s_ = observation_adapter(next_state[AGENT_ID])
                done = done[AGENT_ID]
                r = reward_adapter(next_state[AGENT_ID], a, done)
       
                s = s_
                if done:
                    s = env.reset()                   
                    reward_total = 0
                    error = 0
                    cumulate_flag = True
                    print("wtf?")
                    break
                continue
       
            ##### Select and perform an action ######
            if mode == 'driver': 
                _, a = agent.select_action(s)
            else:
                _, a = agent.select_action_deterministic(s)
       
            action = {AGENT_ID:action_adapter(a)}
            next_state, reward, done, info = env.step(action)
            s_ = observation_adapter(next_state[AGENT_ID])
            done = done[AGENT_ID]
            r = reward_adapter(next_state[AGENT_ID], a, done)
       
            if done and not info[AGENT_ID]['env_obs'].events.reached_goal:
                r -= 1.0
       
            lane_name = info[AGENT_ID]['env_obs'].ego_vehicle_state.lane_id
            lane_id = info[AGENT_ID]['env_obs'].ego_vehicle_state.lane_index
       
            ##### Preprocessing ######
            state, action, reward, next_state = preprocess(s, a, r, s_)
       
            reward_total += reward                    
            s = s_
       
            if done:
                if info[AGENT_ID]['env_obs'].events.reached_goal:
                    success += 1
                
                print('\n|Epoc:', ep,
                      '\n|Step:', t,
                      '\n|Collision:', bool(len(info[AGENT_ID]['env_obs'].events.collisions)),
                      '\n|Off Road:', info[AGENT_ID]['env_obs'].events.off_road,
                      '\n|Goal:', info[AGENT_ID]['env_obs'].events.reached_goal,
                      '\n|Off Route:', info[AGENT_ID]['env_obs'].events.off_route,
                      '\n|R:', reward_total,
                      '\n|Algo:', name,
                      '\n|seed:', seed,
                      '\n|Env:', env_name)
       
                break
            
        ep += 1
        avg_reward_list.append(reward_total)
        print("\n..............................................")
        print("%i Loop, Steps: %i, Avg Reward: %f, Success No. : %i " % (ep, t, reward_total, success))
        print("..............................................")

    reward = statistics.mean(avg_reward_list)
    print("\n..............................................")
    print("Average Reward over %i Evaluation Episodes, At Epoch: %i, Avg Reward:%f, Success No.: %i" % (eval_episodes, ep, reward, success))
    print("..............................................")
    return reward

# observation space
def observation_adapter(env_obs):
    global states

    new_obs = env_obs.top_down_rgb[1]# / 255.0
    states[:, :, 0:3] = states[:, :, 3:6]
    states[:, :, 3:6] = states[:, :, 6:9]
    states[:, :, 6:9] = new_obs
    ogm = env_obs.occupancy_grid_map[1] 
    drivable_area = env_obs.drivable_area_grid_map[1]

    if env_obs.events.collisions or env_obs.events.reached_goal:
        states = np.zeros(shape=(screen_size, screen_size, 9))

    return np.array(states, dtype=np.uint8)

# reward function
def reward_adapter(env_obs, action, done, engage=False):
    ego_obs = env_obs.ego_vehicle_state
    ego_lat_error = ego_obs.lane_position.t
    ego_speed = env_obs.ego_vehicle_state.speed
    lane_name = ego_obs.lane_id
    lane_id = ego_obs.lane_index

    if env_name == 'merge':
        if lane_name == 'gneE6_0' and action > 1:
        # if lane_name == 'gneE6_0' and action == 2:
            off_road = - 1.0
            print('Off lane at junction')
        elif (lane_name == 'gneJ5_3_0' or lane_name == 'gneE4_0') and action == 2:
            off_road = - 1.0
            print('off lane at link, right turn!')
        elif lane_name == 'gneE4_2' and action == 3:
            off_road = - 1.0
            print('off lane at highway, left turn!')
        else:
            off_road = 0.0

        if lane_name == 'gneE4_2':
            target_lane = 0.2
        elif lane_name == 'gneE4_0':
            target_lane = - 0.4
        elif lane_name == 'gneE4_1':
            target_lane = - 0.4
        else:
            target_lane = 0.0
        
        heuristic = ego_speed * 0.002

    elif env_name =='leftturn':
        if lane_name == 'E0_0' and action > 1:
            off_road = - 1.0
            print('Off lane at E0_0')
        elif lane_name == 'E1_0' and ego_lat_error < 0.0 and action == 2:
            off_road = - 1.0
            print('Off lane at E1_0')
        elif lane_name == 'E1_1' and ego_lat_error > 0.0 and action == 3:
            off_road = - 1.0
            print('Off lane at E1_1')
        else:
            off_road = 0.0
        
        heuristic = ego_speed * 0.002 if ego_speed > 2.0 else - 0.05
        
    else:
        off_road = 0.0
        target_lane = 0.0
        heuristic = env_obs.ego_vehicle_state.speed * 0.002

    if done:
        if env_obs.events.reached_goal:
            print('\n Goal')
            goal = 2.0
        else:
            goal = -2.0
    else:
        goal = 0.0

    if env_obs.events.collisions:
        print('\n crashed')
        crash = -2.0
    else:
        crash = 0.0
        
    if env_obs.events.on_shoulder:
        print('\n on_shoulder')
        performance = - 0.04
    else:
        performance = 0.0
        
    if action > 0:
        penalty = - 0.02
    else:
        penalty = 0.0
        
    if engage and PENALTY_GUIDANCE:
        guidance = - 0.1
    else:
        guidance = 0.0

    return heuristic + off_road + goal + crash + performance +\
           guidance + penalty

# action space
def action_adapter(model_action): 

    # discretization
    if model_action == 0:
        lane = 'keep_lane'
    elif model_action == 1:
        lane = 'slow_down'
    elif model_action == 2:
        lane = 'change_lane_right'
    else:
        lane = 'change_lane_left'

    return lane

# information
def info_adapter(observation, reward, info):
    return info

def interaction(COUNTER):
    cumulate_flag = True
    save_threshold = 0.0
    trigger_reward = 0.0 #10.0
    trigger_epoc = 200
    saved_epoc = 1
    goal_counter = 0
    driver_ensemble = []
    entropy_driver_list = deque(maxlen=10)
    variance_driver_list = deque(maxlen=50)

    ##### Human-in-the-Loop #####
    if mode != 'DRL' and mode != 'PGDQN':
        directory = './driver_model/' + scenario     
        for idx in range(1, 11):
            filename = 'driver_RKL_' + str(idx)
            driver = DriverModel(channel, n_actions, seed)
            driver.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, filename)))            
            driver.eval()
            driver.to(device)
            driver_ensemble.append(driver)
    
    for epoc in tqdm(range(1, MAX_NUM_EPOC+1), ascii=True):
        reward_total = 0.0 
        error = 0.0 

        obs = env.reset()
        s = observation_adapter(obs[AGENT_ID])
        
        guidance_count = int(0)
        guidance_rate = 0.0
        frame_skip = 3
        
        for t in count():
            
            if t > MAX_NUM_STEPS:
                print('Max Steps Done.')
                break
            
            ##### Skip first several frames #####
            if t <= frame_skip:
                ###### Select and perform an action #######
                distribution, a = agent.select_action(s, epoc)
                action = {AGENT_ID:action_adapter(a)}
                engage = int(0)
                next_state, reward, done, info = env.step(action)
                s_ = observation_adapter(next_state[AGENT_ID])
                done = done[AGENT_ID]
                r = reward_adapter(next_state[AGENT_ID], a, done, engage)
                s = s_
                if done:
                    s = env.reset()                   
                    reward_total = 0
                    error = 0
                    cumulate_flag = True
                    print("wtf?")
                    break
                continue

            ##### Select and perform an action ######
            distribution, rl_a = agent.select_action(s, epoc)
            entropy_rl = scipy.stats.entropy(distribution)
            guidance = False
            
            ###### Heterogeneous Human Intervention ######
            if epoc >= THRESHOLD and epoc % 2 == 0 and HUMAN_GUIDENCE:
                driver_mean, driver_variance, probability, driver_a =\
                    agent.driver_decision(s, driver_ensemble, online=True)
                variance_driver_list.append(driver_variance[driver_a])
                entropy_driver = scipy.stats.entropy(driver_mean.cpu().numpy())
                entropy_driver_list.append(entropy_driver)

                ###### Intervention Guardian ######
                if action_adapter(rl_a) != action_adapter(driver_a):
                    coeff = - 3.0 / scipy.stats.entropy(np.ones(n_actions)/n_actions) * entropy_rl + 3.0
                    mean_driver = np.mean(entropy_driver_list)
                    std_driver = np.std(entropy_driver_list)
                    threshold = mean_driver + coeff * std_driver
                    
                    if entropy_rl > threshold:
                        guidance = True
            else:
                driver_mean = 0.0
                driver_variance = 0.0
            
            ###### Assign final action ######
            if guidance:
                a = driver_a
                engage = int(1)
                guidance_count += int(1)
            else:
                a = rl_a
                engage = int(0)
            
            ##### Interaction #####
            action = {AGENT_ID:action_adapter(a)}
            next_state, reward, done, info = env.step(action)
            s_ = observation_adapter(next_state[AGENT_ID])
            done = done[AGENT_ID]
            r = reward_adapter(next_state[AGENT_ID], a, done, engage)

            ##### Preprocessing ######
            state, action, reward, next_state = preprocess(s, a, r, s_)
            ##### Store the transition in memory ######
            agent.store_transition(state, action, reward, next_state, engage, done)
            reward_total += reward
    
            if epoc >= THRESHOLD:
                
                if agent.preference and t % PREFERENCE_FREQ == 0:
                    agent.optimize_preference_network(s, a, reward, s_, engage, driver_mean,
                                                 driver_variance, variance_driver_list)
                
                if t % Q_FREQ == 0:
                    agent.optimize_Q_network(s, a, reward, s_, driver_ensemble,
                                         variance_driver_list)

    
            if epoc >= THRESHOLD and done:
                if epoc % 2 == 0:
                    guidance_rate = guidance_count / (t - frame_skip) * 100
                    train_durations.append(epoc)
                    guidance_list.append(guidance_rate)

                goal_list.append(goal_counter/(epoc))                

                agent.scheduler.step()
                if (agent.preference and not agent.together):
                    agent.scheduler_p.step()
                    
            s = s_
    
            if done:
                
                if info[AGENT_ID]['env_obs'].events.reached_goal:
                    goal_counter += 1
                    print("Reached Goal! Goal_Counter:", goal_counter, " Epoc:", epoc)

                reward_list.append(reward_total)
                reward_mean_list.append(np.mean(reward_list[-20:]))
                
                ###### Evaluating the performance of current model ######
                if reward_mean_list[-1] >= trigger_reward and epoc > trigger_epoc:
                    trigger_reward = reward_mean_list[-1]
                    print("Evaluating the Performance.")
                    avg_reward = evaluate(agent, EVALUATION_EPOC)
                    if avg_reward > save_threshold:
                        print('Save the model at %i epoch, reward is: %f' % (epoc, avg_reward))
                        saved_epoc = epoc
                        torch.save(agent.policy_net.state_dict(), os.path.join('trained_network/'+ scenario,
                                  name+'_memo'+str(MEMORY_CAPACITY)+'_epoc'+
                                  str(MAX_NUM_EPOC)+'_seed'
                                  + str(seed)+'_' + str(agent.lr_p)+'_'+env_name+'_policynet.pkl'))
                        save_threshold = avg_reward

                print('\n|Epoc:', epoc,
                      '\n|Step:', t,
                      '\n|Goal Rate:', goal_list[-1],
                      '\n|Goal:', info[AGENT_ID]['env_obs'].events.reached_goal,
                      '\n|Guidance Rate:', guidance_rate, '%',
                      '\n|Collision:', bool(len(info[AGENT_ID]['env_obs'].events.collisions)),
                      '\n|Off Road:', info[AGENT_ID]['env_obs'].events.off_road,
                      '\n|Off Route:', info[AGENT_ID]['env_obs'].events.off_route,
                      '\n|ExpR:', agent.eps_threshold,
                      '\n|Temperature:', agent.temperature_copy,
                      '\n|R:', reward_total,
                      '\n|Algo:', name,
                      '\n|seed:', seed,
                      '\n|Env:', env_name)
    
                s = env.reset()
                reward_total = 0
                error = 0
                cumulate_flag = True
                break
        
        if epoc % PLOT_INTERVAL == 0:
            plot_animation_figure(saved_epoc)
    
        if (epoc % SAVE_INTERVAL == 0):
            np.save(os.path.join('store/' + scenario, 'reward_memo'+str(MEMORY_CAPACITY)+
                                  '_epoc'+str(MAX_NUM_EPOC)+'_seed'+ str(seed) +
                                  '_' + str(agent.lr_p)+'_'+env_name+'_' + name),
                    [reward_mean_list], allow_pickle=True, fix_imports=True)
    
            np.save(os.path.join('store/' + scenario, 'steps_memo'+str(MEMORY_CAPACITY)+
                                  '_epoc'+str(MAX_NUM_EPOC)+'_seed'+ str(seed) +
                                  '_' + str(agent.lr_p)+'_'+env_name+'_' + name),
                    [train_durations], allow_pickle=True, fix_imports=True)   
            
            np.save(os.path.join('store/' + scenario, 'engage_memo'+str(MEMORY_CAPACITY)+
                                  '_epoc'+str(MAX_NUM_EPOC)+'_seed'+ str(seed) +
                                  '_' + str(agent.lr_p)+'_'+env_name+'_' + name),
                    [guidance_list], allow_pickle=True, fix_imports=True)       
                    
    print('Complete')
    return save_threshold

if __name__ == "__main__":

    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings(action="ignore", message="unclosed", category=ResourceWarning)

    plt.ion()
    
    path = os.getcwd()
    yaml_path = os.path.join(path, 'config.yaml')
    with open(yaml_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    ##### Individual parameters for each model ######
    mode = 'UnaHug'
    mode_param = config[mode]
    name = mode_param['name']
    PREFERENCE = mode_param['PREFERENCE']
    DOUBLE = mode_param['DOUBLE']
    DUELING = mode_param['DUELING']
    HUMAN_GUIDENCE = mode_param['HUMAN_GUIDENCE']
    POLICY_GUIDANCE = mode_param['POLICY_GUIDANCE']
    VALUE_GUIDANCE = mode_param['VALUE_GUIDANCE']
    PENALTY_GUIDANCE = mode_param['PENALTY_GUIDANCE']
    ADAPTIVE_CONFIDENCE = mode_param['ADAPTIVE_CONFIDENCE']

    ###### Default parameters for DRL ######
    IMPORTANTSAMPLING  = config['IMPORTANTSAMPLING']
    ENTROPY = config['ENTROPY']
    TOGETHER = config['TOGETHER']
    THRESHOLD = config['THRESHOLD']
    TARGET_UPDATE = config['TARGET_UPDATE']
    BATCH_SIZE = config['BATCH_SIZE']
    GAMMA = config['GAMMA']
    MEMORY_CAPACITY = config['MEMORY_CAPACITY']
    PREFERENCE_FREQ = config['PREFERENCE_FREQ']
    Q_FREQ = config['Q_FREQ']
    EPS_START = config['EPS_START']
    EPS_END = config['EPS_END']
    EPS_DECAY= config['EPS_DECAY']
    FRAME_HISTORY_LEN = config['FRAME_HISTORY_LEN']
    MAX_NUM_EPOC = config['MAX_NUM_EPOC']
    MAX_NUM_STEPS = config['MAX_NUM_STEPS']
    PLOT_INTERVAL = config['PLOT_INTERVAL']
    SAVE_INTERVAL = config['SAVE_INTERVAL']
    DECISION_VARIABLE = config['DECISION_VARIABLE']
    EVALUATION_EPOC = config['EVALUATION_EPOC']

    #### Environment specs ####
    env_name = config['env_name']
    
    if env_name == 'leftturn':
        scenario = 'LeftTurn'
    else:
        scenario = 'RampMerge'
    
    if not os.path.exists("./store/" + scenario):
        os.makedirs("./store/" + scenario)

    if not os.path.exists("./trained_network/" + scenario):
        os.makedirs("./trained_network/" + scenario)

    screen_size = config['screen_size']
    view = config['view']
    AGENT_ID = config['AGENT_ID']
    ACTION_SPACE = gym.spaces.Discrete(DECISION_VARIABLE)
    OBSERVATION_SPACE = gym.spaces.Box(low=0, high=1, shape=(screen_size, screen_size, 9))
    states = np.zeros(shape=(screen_size, screen_size, 9))

    ##### Define agent interface #######
    agent_interface = AgentInterface(
        max_episode_steps=MAX_NUM_STEPS,
        road_waypoints=True,
        neighborhood_vehicle_states=NeighborhoodVehicles(radius=100),
        top_down_rgb=RGB(screen_size, screen_size, view/screen_size),
        occupancy_grid_map=OGM(screen_size, screen_size, view/screen_size),
        drivable_area_grid_map=DrivableAreaGridMap(screen_size, screen_size, view/screen_size),
        action=ActionSpaceType.Lane,
    )
    
    ###### Define agent specs ######
    agent_spec = AgentSpec(
        interface=agent_interface,
        observation_adapter=observation_adapter,
        reward_adapter=reward_adapter,
        action_adapter=action_adapter,
        info_adapter=info_adapter,
    )
    
    legend_bar = []
    seed_list = [97,98,99]
    ##### Train #####
    for i in range(0, 3):
        seed = seed_list[i]
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        ##### Create Env ######
        scenario_path = ['Scenario/' + str(scenario)]
        env = HiWayEnv(scenarios=scenario_path, agent_specs={AGENT_ID: agent_spec},
                       headless=False, visdom=False, sumo_headless=True, seed=seed)
        env.observation_space = OBSERVATION_SPACE
        env.action_space = ACTION_SPACE
        env.agent_id = AGENT_ID

        obs = env.reset()
        img_h, img_w, channel = screen_size, screen_size, 9
        p_state = 1
        n_obs = img_h * img_w * channel
        n_actions = env.action_space.n
        
        # create RL agents
        agent = DQN(img_h, img_w, channel, p_state, n_obs, n_actions, DOUBLE, DUELING,
                    IMPORTANTSAMPLING, PREFERENCE, ENTROPY, TOGETHER, BATCH_SIZE,
                    GAMMA, EPS_START, EPS_END, EPS_DECAY, THRESHOLD, MEMORY_CAPACITY, seed)
        
        agent.policy_guidance = POLICY_GUIDANCE
        agent.value_guidance = VALUE_GUIDANCE
        agent.adaptive_weight = ADAPTIVE_CONFIDENCE
        
        legend_bar.append('seed'+str(seed))
        
        train_durations = []
        train_durations_mean_list = []
        reward_list = []
        reward_mean_list = []
        goal_list = []
        goal_list.append(0.0)
        guidance_list = []

        
        print('\nThe object is:', mode, '\n|Prefrence:', agent.preference,
             '\n|Double:', agent.double, '\n|Dueling:', agent.dueling,
             '\n|Together:', agent.together, '\n|Seed:', agent.seed, 
             '\n|VALUE_GUIDANCE:', VALUE_GUIDANCE, '\n|PENALTY_GUIDANCE:', PENALTY_GUIDANCE,'\n')
        
        success_count = 0

        save_threshold = interaction(success_count)
        
        np.save(os.path.join('store/' + scenario, 'reward_memo'+str(MEMORY_CAPACITY)+
                                  '_epoc'+str(MAX_NUM_EPOC)+'_seed'+ str(seed) +
                                  '_' + str(agent.lr_p)+'_'+env_name+'_' + name),
                [reward_mean_list], allow_pickle=True, fix_imports=True)
        
        np.save(os.path.join('store/' + scenario, 'steps_memo'+str(MEMORY_CAPACITY)+
                                  '_epoc'+str(MAX_NUM_EPOC)+'_seed'+ str(seed) +
                                  '_' + str(agent.lr_p)+'_'+env_name+'_' + name),
                [train_durations], allow_pickle=True, fix_imports=True)        
        
        np.save(os.path.join('store/' + scenario, 'engage_memo'+str(MEMORY_CAPACITY)+
                                  '_epoc'+str(MAX_NUM_EPOC)+'_seed'+ str(seed) +
                                  '_' + str(agent.lr_p)+'_'+env_name+'_' + name),
                [guidance_list], allow_pickle=True, fix_imports=True)     

        print("Evaluating the Performance.")
        avg_reward = evaluate(agent, EVALUATION_EPOC)
        if avg_reward > save_threshold:
            print('Save the model!')
            torch.save(agent.policy_net.state_dict(), os.path.join('trained_network/' + scenario,
                      name+'_memo'+str(MEMORY_CAPACITY)+'_epoc'+
                      str(MAX_NUM_EPOC)+'_seed'
                      + str(seed)+'_' + str(agent.lr_p)+'_'+env_name+'_policynet.pkl'))

        env.close()