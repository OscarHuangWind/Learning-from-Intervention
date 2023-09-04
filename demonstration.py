#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 14:45:40 2023

@author: oscar
"""

import os
import gym
import torch
import numpy as np
from time import sleep
import matplotlib.pyplot as plt

from smarts.core.agent_interface import NeighborhoodVehicles, RGB
from smarts.core.agent_interface import AgentInterface
from smarts.core.controllers import ActionSpaceType
from smarts.zoo.agent_spec import AgentSpec
from smarts.env.hiway_env import HiWayEnv
from pynput.keyboard import Key, Listener

#### Logitech G29 ######
import pygame
from g29 import Controller

screen_size = 120
n_action = 4
view = 50 #32
ACTION_SPACE = gym.spaces.Discrete(n_action)
OBSERVATION_SPACE = gym.spaces.Box(low=0, high=1, shape=(screen_size, screen_size, 9))

def plot_animation_figure():
    plt.figure()
    plt.clf()
    plt.subplot(2, 1, 1)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.plot(reward_list)
    plt.plot(reward_mean_list)

    plt.subplot(2, 1, 2)
    plt.xlabel('Episode')
    plt.ylabel('Goal')
    plt.plot(goal_list)

    plt.pause(0.001)
    plt.tight_layout()
    plt.show()

def reward_adapter(env_obs, action, done, engage=False):
    ego_obs = env_obs.ego_vehicle_state
    ego_lat_error = env_obs.ego_vehicle_state.lane_position.t
    ego_speed = env_obs.ego_vehicle_state.speed
    lane_name = ego_obs.lane_id
    lane_id = ego_obs.lane_index

    if env_name == 'RampMerge':
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

    elif env_name =='LeftTurn':
        if lane_name == 'E0_0' and action > 1:
            off_road = - 1.0
            print('Off lane at E0_0')
        elif lane_name == 'E1_0' and ego_lat_error < - 0.1 and action == 2:
            off_road = - 1.0
            print('Off lane at E1_0')
        else:
            off_road = 0.0

        target_lane = 0.0
        heuristic = ego_speed * 0.002 if ego_speed > 1.0 else - 0.05
        
    else:
        off_road = 0.0
        target_lane = 0.0
        heuristic = ego_speed * 0.002

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

    return heuristic + off_road + goal + crash + performance + penalty + target_lane

class HumanKeyboardPolicy():
    def __init__(self):
        # initialize the keyboard listener
        self.listener = Listener(on_press=self.on_press)
        self.listener.start()

        # initialize desired speed and lane
        self.action = 0
        self.command = None

    def on_press(self, key):
        """To control, use the keys:
        Up: to speed up
        Down: to slow down
        Left: to change left
        Right: to change right
        """

        if key == Key.up:
            self.action = 0
        elif key == Key.down:
            self.action = 1
        elif key == Key.right:
            self.action = 2
        elif key == Key.left:
            self.action = 3

    def act(self, obs):
        command = 'keep_lane'
        if self.action == 0:
            command = 'keep_lane'
        elif self.action == 1:
            command = 'slow_down'
        elif self.action == 2:
            command = 'change_lane_right'
        elif self.action == 3:
            command = 'change_lane_left'

        return self.action, command

def HumanDriverControl(controller):
    pygame.event.pump()
    steerPos = controller.get_steer()
    if abs(steerPos) < 0.04:
        steerPos = None
    
    throtPos = (-controller.get_throttle() + 1.0)/2
    breakPos = (-controller.get_break() + 1.0)/2

    if breakPos > 0.05:
        command = 'slow_down'
        action = 1
    elif steerPos is not None:
        if steerPos >= 0.0:
            command = 'change_lane_right'
            action = 2
        else:
            command = 'change_lane_left'
            action = 3
    else:
        command = 'keep_lane'
        action = 0
    
    # if throtPos < 0.02:
    #     command = 'slow_down'
    #     action = 1
    # elif steerPos is not None:
    #     if steerPos >= 0.0:
    #         command = 'change_lane_right'
    #         action = 2
    #     else:
    #         command = 'change_lane_left'
    #         action = 3
    # # elif throtPos > 0.0:
    # else:
    #     command =  'keep_lane'
    #     action = 0
        
    return action, command

def main(scenario, num_episodes, seed):
    max_num_steps = 400
    agent_spec = AgentSpec(
        interface = AgentInterface(
            max_episode_steps=max_num_steps,
            road_waypoints=True,
            neighborhood_vehicle_states=NeighborhoodVehicles(radius=100),
            top_down_rgb=RGB(screen_size, screen_size, view/screen_size),
            action=ActionSpaceType.Lane,
        ),
        agent_builder = HumanKeyboardPolicy,
        agent_params=None,
    )

    if scenario == 'RampMerge':
        scenario_path = ['Scenario/RampMerge']
    elif scenario == 'LeftTurn':
        scenario_path = ['Scenario/LeftTurn']
    else:
        print('No such a scenario!')
        return

    env = HiWayEnv(scenarios=scenario_path, agent_specs={driver: agent_spec}, 
                   sumo_headless=True, headless=False, seed=seed)

    env.seed = seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # build agent
    agent = agent_spec.build_agent()

    episode = 1
    goal_counter = 0
    ep = 0
    
    controller = Controller(0)

    while True:
        # data collection
        obs = []
        act = []
        reward_total = 0.0

        # start env
        observation = env.reset()
        observation = observation[driver]
        done = False
        states = np.zeros(shape=(screen_size, screen_size, 9))

        while not done:

            agent_obs = observation
            states[:, :, 0:3] = states[:, :, 3:6]
            states[:, :, 3:6] = states[:, :, 6:9]
            states[:, :, 6:9] = agent_obs.top_down_rgb[1] / 255.0
            obs.append(states.astype(np.float32))            

            ##### Logitech G29 ######
            human_action, human_command = HumanDriverControl(controller)

            ###### Keyboard Control ######
            # human_action, human_command = agent.act(agent_obs)
            
            if agent_obs.ego_vehicle_state.lane_id == 'E0_0' and human_action > 1:
                human_action = 0
                human_command = 'keep_lane'
                print('Please initial your steering wheel!!!!')
            
            observation, _, done, info = env.step({driver: human_command})
            observation = observation[driver]
            done = done[driver]
            r = reward_adapter(observation, human_action, done)
            reward_total += r
            act.append(human_action)
            
            sleep(1/15)
        
        if info[driver]['env_obs'].events.reached_goal:
            
            #### For G29 #####
            if controller.abandon_flag:
                np.savez('expert_data/{}/{}/demo_{}_abandon.npz'.format(scenario, driver, episode),
                          obs=np.array(obs, dtype=np.float32), act=np.array(act, dtype=np.float32))
            elif controller.corner_flag:
                np.savez('expert_data/{}/{}/demo_{}_corner.npz'.format(scenario, driver, episode),
                          obs=np.array(obs, dtype=np.float32), act=np.array(act, dtype=np.float32))
            else:
                np.savez('expert_data/{}/{}/demo_{}.npz'.format(scenario, driver, episode),
                          obs=np.array(obs, dtype=np.float32), act=np.array(act, dtype=np.float32))
            
            #### For Keyboard Control #####
            # np.savez('expert_data/{}/{}/demo_{}.npz'.format(scenario, driver, episode),
            #          obs=np.array(obs, dtype=np.float32), act=np.array(act, dtype=np.float32))
                
            episode += 1
            goal_counter += 1
        
        ep += 1
        goal_list.append(goal_counter/ep)
        reward_list.append(reward_total)
        reward_mean_list.append(np.mean(reward_list[-20:]))

        controller.corner_flag = False
        controller.abandon_flag = False

        print('\n|Epoc:', ep,
              '\n|R:', reward_total,
              '\n|Goal Rate:', goal_list[-1],
              '\n|Goal:', info[driver]['env_obs'].events.reached_goal,
              '\n|Collision:', bool(len(info[driver]['env_obs'].events.collisions)),
              '\n|Off Road:', info[driver]['env_obs'].events.off_road,
              '\n|Off Route:', info[driver]['env_obs'].events.off_route,
              '\n|Env:', env_name)

        # if ep % 10 == 0:
        #     plot_animation_figure()
        
        if episode > num_episodes:
            break

    env.close()


if __name__ == "__main__":
    # scenario = 'RampMerge'
    scenario = 'LeftTurn'
    env_name = scenario
    
    seed = 6
    
    driver = 'oscar_seed' + str(seed)
    samples = 40
    reward_list = []
    reward_mean_list = []
    goal_list = []
    
    if not os.path.exists('./expert_data/{}/{}'.format(scenario, driver)):
        os.makedirs('./expert_data/{}/{}'.format(scenario, driver))

    pygame.init()
    
    clock = pygame.time.Clock()

    main(scenario=scenario, num_episodes=samples, seed=seed)#random.randint(0, 100))
    
    pygame.quit()