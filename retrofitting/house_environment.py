# from gym import Env, spaces
import gymnasium as gym
from gymnasium import Env, spaces
import numpy as np
import pandas as pd
from gamma_deterioration_copy_copy import matrices_gen 
import random

class House(Env):
    def __init__(self, house_size_m2: float = 120):
        super(House, self).__init__()

        ###################CLASS ATTRIBUTES######################
        self.current_state = 0
        self.time = 0
        self.num_years = 60
        self.time_step = 5
        self.state_space = House.get_state_space(num_damage_states=3,num_years= self.num_years, time_step= self.time_step) 
        self.num_states = len(self.state_space)

        num_actions = 4
        ##### ACTIONS #####
        # 0,  # DO_NOTHING
        # 1,  # FIX_ROOF
        # 2,  # FIX_WALL
        # 3   # FIX_FACADE
        self.action_space = spaces.Discrete(num_actions)
        self.observation_space = spaces.Discrete(self.num_states)
        self.p ,self.change_matrix = self.get_transition_matrices(saved_version=True,simple_stuff=True)
        self.house_size_m2 = house_size_m2
        self.kwh_per_state = self.import_simulation_data(file_path='building_scenarios_copy.csv',no_windows=True)
        self.renovation_costs = np.array([0, 2000, 5000, 3000])    # [cost_doNothing, cost_roof, cost_wall, cost_cellar] # TODO: should change according to m2
        self.energy_demand_nominal = [57, 95, 38]  # [roof, wall, cellar]
        self.degradation_rates = [0.0, 0.2, 0.4]
        self.probability_matrices,self.action_matrices = self.get_transition_matrices(saved_version = True, simple_stuff= True)
        self.probability_matrices = self.state_space_probability(self.probability_matrices,self.state_space, save_version =True, import_version=False)
        self.ages = [0,0,0]
        ###################FUNCTIONS######################

    def one_hot_enc(self,action):
        # Get transition probabilities, next state, and reward
        action_one_hot_enc = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
        act = action_one_hot_enc[action]  
        return act
    
    def import_simulation_data(self,file_path, no_windows: bool):
        data = pd.read_csv(file_path) 
        list = []
        if no_windows:
            val = 3
        else: 
            val = 1
        for indx in range(0,len(data),val):
            kwh_per_m2 = data['energy[kWh/m2]'][indx]
            list.append(kwh_per_m2)  
        return list

    @staticmethod
    def get_state_space(num_damage_states: int, num_years: int, time_step: int):
        state_space = {}
        state_number = 0
        for r_damage_state in range(num_damage_states):
            for w_damage_state in range(num_damage_states):
                for c_damage_state in range(num_damage_states):
                    for age_r in range(0,num_years+1,time_step):
                        for age_w in range(0,num_years+1,time_step):
                            for age_f in range(0,num_years+1,time_step):
                                state_space[state_number] = (r_damage_state, w_damage_state, c_damage_state,age_r,age_w,age_f)
                                state_number += 1
        return state_space

    def get_transition_matrices(self, saved_version: bool, simple_stuff: bool):
        if saved_version:
            p = np.load('transition_matrices_trial.npy')
        else: 
            p = matrices_gen(SIMPLE_STUFF = simple_stuff,N= 1000 ,T = self.num_years, n = self.time_step)
        # Round all components of the array to three decimal places
        p = np.round(p, 3)
        # Define the transition matrices for action
        transition_matrix = np.array([[1, 0, 0],
                                    [1, 0, 0],
                                    [1, 0, 0]])
        return p,transition_matrix

    @staticmethod
    def energy2euros(num_of_kwh: float) -> float:
        price_kwh = 0.35
        return price_kwh * num_of_kwh

    def get_reward(self, action: int, current_state: int) -> float:
        action_costs = self.renovation_costs[action]
        total_energy_demand = self.kwh_per_state[current_state]
        energy_bills = House.energy2euros(total_energy_demand)
        energy_bills = energy_bills * self.house_size_m2
        net_cost = action_costs + energy_bills
        reward = -net_cost
        return reward
    
    def state_space_probability(self, probability_matrices,state_space, save_version:bool, import_version:bool):
        if import_version :
            overall_probability = np.load("age_matrices.npy")
        else:
            # overall_probability = np.zeros((len(state_space),len(state_space)))
            overall_probability = []
            # Iterate over all possible next states
            for current_state in range(len(state_space)):
                list2 = []
                st1 = state_space[current_state][0]
                age1 = state_space[current_state][3]
                st2 = state_space[current_state][1]
                age2 = state_space[current_state][4]
                st3 = state_space[current_state][2]
                age3 = state_space[current_state][5]
                for next_state in range(len(state_space)):
                    st4 = state_space[next_state][0]
                    age4 = state_space[next_state][3]
                    st5 = state_space[next_state][1]
                    age5 = state_space[next_state][4]
                    st6 = state_space[next_state][2] # TODO check to change the ages to the futures ages
                    age6 = state_space[next_state][5]
                    prob1 = probability_matrices[age4][st1][st4]
                    prob2 = probability_matrices[age5][st2][st5]
                    prob3 = probability_matrices[age6][st3][st6]
                    lst = [prob1,prob2,prob3]
                    lst = np.array(lst)
                    lst = np.prod(lst)  
                    list2.append(int(lst))
                list2 = np.array(list2)      
                overall_probability.append(list2)
            overall_probability = np.array(overall_probability)
            if save_version :
                np.save("age_matrices.npy", overall_probability)    
            return overall_probability
            
    def get_transition_probs(self, current_state: int, action: int):
        """
        Function that calculates probabilities.
        Parameters
        ----------
        current_state : int
            The current state index.
        action : int
            The action index.
        time : int
            The current time in the episode.
        Returns
        -------
        transition_probs : array
            The transition probabilities for next states.
        next_state : int
            The next state index.
        reward : float
            The reward from taking the action.
        """
        transition_probabilities = []
        pr = self.calculate_system_probability(current_state = current_state, probability_matrices = self.p,state_space=self.state_space, action = action)
        for next_state in range(self.num_states):
            # prob = self.state_transition_model[action][current_state][next_state]
            prob = pr[next_state]
            # prob = self.calculate_transition_probability(state =current_state,next_state =next_state,ages=self.ages,p = p,action = action,state_space=self.state_space)
            reward = self.get_reward(action=action, current_state=current_state)
            transition_probabilities.append((prob, next_state, reward))
        total_sum = sum(prob for prob, _, _ in transition_probabilities)

        def normalize_array(arr):
            # Step 1: Calculate the sum of all values in the array
            total_sum = sum(arr)
            
            # Step 2 & 3: Normalize each value in the array
            normalized_arr = [value / total_sum for value in arr]
            
            return normalized_arr

        array = np.array(transition_probabilities)
        # Normalize the array
        for i in array:
            normalized_arr = normalize_array(array)

        print("Original array:", array)
        print("Normalized array:", normalized_arr)
        print("Sum of normalized array:", sum(normalized_arr))  # Should be approximately 1


        #Check again
        total_sum = 0
        # Iterate through the list and add each probability to the sum
        for prob, _, _ in transition_probabilities:
            total_sum += prob
        if total_sum != 1:
            raise Exception('Probability is bigger than 1. We guessed wrong')
        return transition_probabilities

    def reset(self, seed = random):
        """
        Resets the environment to its initial state.
        Returns
        -------
        state : int
            The initial state index.
        """
        self.time = 0

        self.current_state = 0
        #reset should have observation
        return 0  # Return initial state index

    def step(self, action):
        """
        Take a step in the environment.
        Parameters
        ----------
        action : int
            The action index.
        Returns
        -------
        state : int
            The next state index.
        reward : float
            The reward from taking the action.
        done : bool
            Whether the episode is done.
        info : dict
            Additional information (unused).
        """ 
        act = self.one_hot_enc(action)
        # Get all transition probabilities for the current state
        transitions = self.get_transition_probs(self.current_state, action)
        # Extract probabilities from transition probabilities
        probabilities = [prob for prob, _, _ in transitions]
        # Choose next state based on transition probabilities
        next_state = np.random.choice(self.num_states, p=probabilities)
        # Calculate state reward
        reward = self.get_reward(action, self.current_state)   
        # Update time
        self.time += self.time_step
        for i in range(len(self.ages)):
            if act[i] == 0:
                self.ages[i] += self.time_step
            else: 
                self.ages[i] = 0
        # Check if episode is done (time limit reached)
        done = self.time >= self.num_years
        self.current_state = next_state
        return next_state, reward, done, {}

    def render(self, mode='human'):
        # Optionally render the environment
        pass

    def close(self):
        # Clean up resources, if any
        pass

