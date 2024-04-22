# from gym import Env, spaces
import gymnasium as gym
from gymnasium import Env, spaces
import numpy as np
import pandas as pd
from gamma_deterioration_copy_copy import matrices_gen 
import random
import matplotlib.pyplot as plt
import time
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO, DQN, A2C
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from sb3_contrib import RecurrentPPO
import torch
import torch as th
from torch import nn
import torch.nn as nn

class House(Env):
    def __init__(self, house_size_m2: float = 120):
        self.current_state = 0
        self.time = 0
        self.num_years = 30
        self.time_step = 15
        self.state_space = House.get_state_space(num_damage_states=3,num_years= self.num_years, time_step= self.time_step) 
        self.num_states = len(self.state_space)



        self.observation_space = spaces.Box(0, 1.25, shape=(self.number_of_panels + 7,))

    def step(self, action):
        '''
        Parameters:

            action (ActType)  an action provided by the agent to update the environment state.
        Returns:

                observation (ObsType)  An element of the environment's observation_space as the next observation due to the agent actions. An example is a numpy array containing the positions and velocities of the pole in CartPole.

                reward (SupportsFloat)  The reward as a result of taking the action.

                terminated (bool)  Whether the agent reaches the terminal state (as defined under the MDP of the task) which can be positive or negative. An example is reaching the goal state or moving into the lava from the Sutton and Barton, Gridworld. If true, the user needs to call reset().

                truncated (bool)  Whether the truncation condition outside the scope of the MDP is satisfied. Typically, this is a timelimit, but could also be used to indicate an agent physically going out of bounds. Can be used to end the episode prematurely before a terminal state is reached. If true, the user needs to call reset().

                info (dict) Contains auxiliary diagnostic information (helpful for debugging, learning, and logging). This might, for instance, contain: metrics that describe the agent’s performance state, variables that are hidden from observations, or individual reward terms that are combined to produce the total reward. In OpenAI Gym <v26, it contains “TimeLimit.truncated” to distinguish truncation and termination, however this is deprecated in favour of returning terminated and truncated variables.

                done (bool)  (Deprecated) A boolean value for if the episode has ended, in which case further step() calls will return undefined results. This was removed in OpenAI Gym v26 in favor of terminated and truncated attributes. A done signal may be emitted for different reasons: Maybe the task underlying the environment was solved successfully, a certain timelimit was exceeded, or the physics simulation has entered an invalid state.


        '''

        return 


    def reset(self):
        '''
                
        Parameters:

                seed (optional int)  The seed that is used to initialize the environment's PRNG (np_random). If the environment does not already have a PRNG and seed=None (the default option) is passed, a seed will be chosen from some source of entropy (e.g. timestamp or /dev/urandom). However, if the environment already has a PRNG and seed=None is passed, the PRNG will not be reset. If you pass an integer, the PRNG will be reset even if it already exists. Usually, you want to pass an integer right after the environment has been initialized and then never again. Please refer to the minimal example above to see this paradigm in action.

                options (optional dict)  Additional information to specify how the environment is reset (optional, depending on the specific environment)

        Returns:

                observation (ObsType)  Observation of the initial state. This will be an element of observation_space (typically a numpy array) and is analogous to the observation returned by step().

                info (dictionary)  This dictionary contains auxiliary information complementing observation. It should be analogous to the info returned by step().


        '''
        pass

    # gymnasium.Env.reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) → tuple[ObsType, dict[str, Any]]



