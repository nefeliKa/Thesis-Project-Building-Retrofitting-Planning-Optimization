from gym import Env, spaces
import numpy as np
import pandas as pd
import pickle 
import csv
from scipy.sparse import lil_matrix, save_npz,load_npz
from gamma_deterioration import matrices_gen 
import time
from scipy.sparse import csr_matrix
from multiprocessing import Pool



class House(Env):
    def __init__(self, house_size_m2: float = 176):
        super(House, self).__init__()
        self.num_damage_states = 3
        self.num_actions = 8
        ##### ACTIONS #####
        # 0,  # DO_NOTHING
        # 1,  # FIX_ROOF
        # 2,  # FIX_WALL
        # 3   # FIX_FACADE
        self.groundfloor_area = 75.40
        # self.groundfloor_area = 40
        self.roof_area = 89.414567
        # self.roof_area = 40
        self.facade_area = 210.80
        # self.facade_area = 45
        self.cost_eps_floor_m2 = 28.91 #euros per m2
        self.cost_eps_facade_m2 = 162.73
        self.cost_eps_roof_m2 = 231.63 
        self.floor_cost = self.groundfloor_area * self.cost_eps_floor_m2
        self.facade_cost = self.facade_area * self.cost_eps_facade_m2
        self.roof_cost = self.roof_area * self.cost_eps_roof_m2
        self.current_state = 0
        self.time = 0
        self.num_years = 60
        self.time_step = 20
        self.action_space = spaces.Discrete(8)
        self.state_space = House.get_state_space(num_damage_states=self.num_damage_states,
                                                 num_years= self.num_years,
                                                 time_step = self.time_step)
        self.num_states = len(self.state_space)
        self.observation_space = spaces.Discrete(self.num_states)

        _, self.kwh_per_state = self.import_simulation_data(file_path='building_scenarios.csv',no_windows=True)
        self.house_size_m2 = house_size_m2

        # # [cost_doNothing, cost_roof, cost_wall, cost_cellar]
        self.renovation_costs = np.array([0, self.roof_cost, self.facade_cost, self.floor_cost,(self.roof_cost+self.facade_cost),(self.roof_cost+self.floor_cost),(self.facade_cost+self.floor_cost),(self.roof_cost+self.facade_cost+self.floor_cost)])  # 

        # self.degradation_rates = [0.0, 0.2, 0.35]
        self.energy_bills = self.get_state_electricity_bill(state_space = self.state_space,kwh_per_state=self.kwh_per_state)
        self.rewards = self.calculate_reward(save= True)
        self.material_probability_matrices,self.action_matrices = \
                                                self.import_gamma_probabilities(calculate_gamma_distribution_probabilities= True,
                                                step_size=self.time_step,SIMPLE_STUFF = True, 
                                                N = 1000, do_plot= True, T = self.num_years+self.time_step,
                                                save_probabilities = True)
        self.health_age_state_space, self.health_age_state_tansition_matrix = self.health_age_state_tansitions(num_damage_states= self.num_damage_states,num_years = self.num_years,material_probability_matrices = self.material_probability_matrices, action_matrices=self.action_matrices,time_step = self.time_step)        # self.state_transition_model = House.get_state_transition_model(num_actions=self.num_actions, state_space=self.state_space, time_step= self.time_step,num_years = self.num_years)
        self.state_transition_model = self.get_state_transition_model(num_actions=self.num_actions,
                                                                      state_space=self.state_space, 
                                                                      time_step= self.time_step,
                                                                      num_years = self.num_years,
                                                                      health_age_state_tansition_matrix =self.health_age_state_tansition_matrix,
                                                                      health_age_state_space= self.health_age_state_space,
                                                                      load_saved_matrices = False)
        # self.transition_probabilities = self.get_transition_probs()    
        # self.rewards = self.calculate_reward(save= True)
 
####################################################################################################

 
#State space as a numpy with ages and time and cleared of all the states that we do not need
    def get_state_space(num_damage_states: int, num_years: int, time_step: int):
            state_space = np.zeros((500000,7))
            state_number = 0
            for time in range(0,(num_years+time_step),time_step):
                for r_damage_state in range(num_damage_states):
                    for w_damage_state in range(num_damage_states):
                        for c_damage_state in range(num_damage_states):
                            for age_r in range(0,num_years+time_step,time_step):
                                for age_w in range(0,num_years+time_step,time_step):
                                    for age_f in range(0,num_years+time_step,time_step):
                                        if time>=age_r and time>=age_w and time>=age_f:
                                            state = np.array([time,r_damage_state,w_damage_state,c_damage_state,age_r,age_w,age_f])
                                            # state_space = np.vstack([state_space, state])
                                            state_space[state_number] = state
                                            # state_space[state_number] = (time,r_damage_state, w_damage_state, c_damage_state,age_r,age_w,age_f)
                                            state_number += 1

            # Delete rows from start_index to the end
            filtered_array = state_space[:state_number]

            
            np.save('state_space.npy',filtered_array)

            return filtered_array
  
    def health_age_state_tansitions(self,num_damage_states:int,num_years:int,material_probability_matrices:np.array, action_matrices:np.array,time_step : int): 
        # state transition of H
        health_probabilities = material_probability_matrices
        # state transitions of A
        len_h = len(health_probabilities)
        time_length = int((num_years+time_step)/time_step)
        age_matrix = np.zeros((time_length,time_length))
        for age in range(1,time_length):
            age_matrix[age-1,age] = 1

        # state trasritions of HA
        length =  int(num_damage_states * time_length)
        t = len(material_probability_matrices)
        health_age_state_space = np.zeros((length, 2))
        #make states
        n = 0
        for health_state in range(num_damage_states):
            for age_state in range(0,time_length):
                health_age_state_space[n] = np.array([health_state,age_state])
                n+= 1
 
        list = []
        for action in range(0,2):
            health_age_state_matrix = np.zeros((len_h,length,length))
            for time in range(0,time_length):
                for  index, state in enumerate(health_age_state_space):
                    for  index2, fstate in enumerate(health_age_state_space):
                        if action == 0:
                            prob_health =  health_probabilities[time,int(state[0]),int(fstate[0])]
                            prob_age = age_matrix[int(state[1]),int(fstate[1])] 
                            health_age_state_matrix[time,index,index2] = prob_health * prob_age
                        else:
                            prob_health =  action_matrices[int(state[0]),int(fstate[0])]

                            prob_age = 1 if int(fstate[1]) == 0 else 0
                            health_age_state_matrix[time,index,index2] = prob_health * prob_age
            list.append(health_age_state_matrix)
        health_age_state_matrix = np.array(list)
        return health_age_state_space,health_age_state_matrix
    
    def get_state_transition_model(self, num_actions: int, state_space: dict, time_step:int, num_years:int,  health_age_state_tansition_matrix:np.array, health_age_state_space:np.array,load_saved_matrices:bool):
        
        if load_saved_matrices: 
            sparse_matrices = []

            # Load each sparse matrix from the saved files
            for i in range(num_actions):
                csr_matrix = load_npz('sparse_matrix_action_{}.npz'.format(i))
                sparse_matrices.append(csr_matrix)

        else: 
            num_damage_states = 3  # good, medium, bad
            action_one_hot_enc = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],[1, 1, 0],[1, 0, 1],[0, 1, 1],[1, 1, 1]])

            # Initialize lists to store sparse matrices for each action
            sparse_matrices = [lil_matrix((len(state_space), len(state_space)), dtype=np.float64) for _ in range(num_actions)]

            for action in range(num_actions):
                print('0')
                for current_state_index, current_state in enumerate(state_space):
                    if current_state_index % 2000 == 0:
                        print(current_state)
                    if current_state[0] == num_years:
                        sparse_matrices[action][current_state_index, current_state_index] = 1
                        continue
                    if action == 0:
                        # Find indices where the first value is equal to current_state[0] + time_step
                        # and values 4, 5, and 6 of state_space are greater than values 4, 5, and 6 of the current state
                        future_states_indices = np.where((state_space[:, 0] == current_state[0] + time_step) &
                                                        (state_space[:, 4] == current_state[4]+time_step) &
                                                        (state_space[:, 5] == current_state[5]+time_step) &
                                                        (state_space[:, 6] == current_state[6]+time_step))[0]

                    elif action == 1: 
                        # Find indices where the first value is equal to current_state[0] + time_step and the fourth value is 0
                        future_states_indices = np.where((state_space[:, 0] == current_state[0] + time_step) & 
                                                        (state_space[:, 4] == 0)&
                                                        (state_space[:, 5] == current_state[5]+time_step)&
                                                        (state_space[:, 6] == current_state[6]+time_step))[0]

                    elif action == 2: 
                        # Find indices where the first value is equal to current_state[0] + time_step and the fourth value is 0
                        future_states_indices = np.where((state_space[:, 0] == current_state[0] + time_step) &
                                                        (state_space[:, 5] == 0)&
                                                        (state_space[:, 4] == current_state[4]+time_step)&
                                                        (state_space[:, 6] == current_state[6]+time_step))[0]

                    elif action == 3: 
                        # Find indices where the first value is equal to current_state[0] + time_step and the fourth value is 0
                        future_states_indices = np.where((state_space[:, 0] == current_state[0] + time_step) &
                                                        (state_space[:, 4] == current_state[4]+time_step) &
                                                        (state_space[:, 5] == current_state[5]+time_step) &
                                                        (state_space[:, 6] == 0))[0]
                    
                    elif action == 4: 
                        # Find indices where the first value is equal to current_state[0] + time_step and the fourth value is 0
                        future_states_indices = np.where((state_space[:, 0] == current_state[0] + time_step) &
                                                        (state_space[:, 4] == 0) &
                                                        (state_space[:, 5] == 0) &
                                                        (state_space[:, 6] == current_state[6]+time_step))[0]
                    elif action == 5: 
                        # Find indices where the first value is equal to current_state[0] + time_step and the fourth value is 0
                        future_states_indices = np.where((state_space[:, 0] == current_state[0] + time_step) &
                                                        (state_space[:, 4] == 0) &
                                                        (state_space[:, 5] ==  current_state[5]+time_step) &
                                                        (state_space[:, 6] == 0))[0]
                    elif action == 6: 
                        # Find indices where the first value is equal to current_state[0] + time_step and the fourth value is 0
                        future_states_indices = np.where((state_space[:, 0] == current_state[0] + time_step) &
                                                        (state_space[:, 4] == current_state[4]+time_step) &
                                                        (state_space[:, 5] == 0) &
                                                        (state_space[:, 6] == 0))[0]
                    elif action == 7: 
                        # Find indices where the first value is equal to current_state[0] + time_step and the fourth value is 0
                        future_states_indices = np.where((state_space[:, 0] == current_state[0] + time_step) &
                                                        (state_space[:, 4] == 0) &
                                                        (state_space[:, 5] == 0) &
                                                        (state_space[:, 6] == 0))[0]
                        
                    # Iterate through the range of future state indices
                    for future_state_index in future_states_indices:
                        future_state = state_space[future_state_index]
                        if future_state[0] == current_state[0] + time_step and current_state[0] != num_years:
                            future_states_probabilities = np.ones(num_damage_states)

                            for i in range(num_damage_states):
                                action_array = action_one_hot_enc[action][i]
                                component_h_a_st = np.array([current_state[i+1], current_state[i+4]//time_step])
                                fcomponent_h_a_st = np.array([future_state[i+1], future_state[i+4]//time_step])
                                component_index_in_state_space = np.where(np.all(health_age_state_space == component_h_a_st, axis=1))[0][0]
                                fcomponent_state_in_state_space = np.where(np.all(health_age_state_space == fcomponent_h_a_st, axis=1))[0][0]
                                probability = health_age_state_tansition_matrix[action_array][int(current_state[i+4]//time_step)][component_index_in_state_space][fcomponent_state_in_state_space]
                                future_states_probabilities[i] = probability

                            new_probability = np.prod(future_states_probabilities)
                            sparse_matrices[action][current_state_index, future_state_index] = new_probability
                        elif current_state[0] == num_years and current_state == future_state:
                             sparse_matrices[action][current_state_index, future_state_index] = 1
                            

            # Save each sparse matrix in the list separately
        # Save each sparse matrix in the list separately
        for i, sparse_matrix in enumerate(sparse_matrices):
            csr_matrix = sparse_matrix.tocsr()  # Convert to CSR format
            save_npz('sparse_matrix_action_{}.npz'.format(i), csr_matrix)
        return sparse_matrices

    def import_gamma_probabilities(self, calculate_gamma_distribution_probabilities:bool,
                                    step_size:int,SIMPLE_STUFF: bool, N :int, do_plot: bool,T:int,
                                    save_probabilities:bool):           
        if calculate_gamma_distribution_probabilities == False:
            load = np.load("transition_matrices.npy")
        else : 
            load = matrices_gen(SIMPLE_STUFF = True, N = N, T = T, do_plot = do_plot,step_size = step_size,save_probabilities=save_probabilities)
        probability_matrices = load 
        action_matrices =np.array([[1., 0., 0.], [1., 0., 0.],[1., 0., 0.]])  
        
        return probability_matrices,action_matrices

    @staticmethod
    def energy2euros(num_of_kwh: float) -> float:
        price_kwh = 0.35
        return price_kwh * num_of_kwh

    def calculate_reward(self,save: bool): 
        print('I am inside rewards')
        if save:
            action_array = np.zeros((self.num_actions,self.num_states))
            for action in range(self.num_actions): 
                for state in range(self.num_states): 
                    action_array[action,state] = self.get_reward(action,state)
            
            np.save('action_array.npy',action_array)
        else: 
            action_array = np.load('action_array.npy')
        return action_array
    
    def get_reward(self, action: int, current_state: int) -> float:
            
            action_costs = self.renovation_costs[action]
            state_name = self.state_space[current_state][1:4]
            total_energy_demand = self.kwh_per_state[tuple(state_name)]
            if total_energy_demand >= 174:
                total_energy_demand = total_energy_demand * 2
            total_energy_demand = total_energy_demand* 250 #multiply by square meters
            energy_bills = House.energy2euros(total_energy_demand)
            if self.state_space[current_state][0] == 0:
                energy_bills = (energy_bills)
            else: 
                energy_bills = (energy_bills)*self.time_step
            # energy_bills = (energy_bills)*self.time_step
            # if total_energy_demand > 210:
            #      energy_bills = 100000
            # if  self.state_space[current_state][0] ==  0: 
            #     net_cost = action_costs + (energy_bills)
            # else:
            #     net_cost = action_costs + (energy_bills*self.time_step)
            net_cost = action_costs + energy_bills
            reward = -net_cost

            # place zero reward to final states /absorbing states  
            if  self.state_space[current_state][0] ==  self.num_years: 
                reward = 0

            return reward 

    def get_state_electricity_bill(self,state_space,kwh_per_state):
        energy_bills = {}
        for current_state in state_space:
            current_state = current_state.astype(int)
            state_name = current_state[1:4]
            total_energy_demand = kwh_per_state[tuple(state_name.astype(int))]
            energy_bills[tuple(current_state)] = total_energy_demand

        return energy_bills

    def import_simulation_data(self,file_path, no_windows: bool):
        data = pd.read_csv(file_path) 
        list = []
        list2 = [] 
        dict = {}
        if no_windows:
            val = 3
        else: 
            val = 1
        for indx in range(0,len(data)):
        # for indx in range(0,len(data),val):
            kwh_per_m2 = data['energy[kWh/m2]'][indx]
            string = data['state'][indx]
            # Remove the parentheses and split the string by commas
            split_string = string.strip("()").split(", ")
            # Extract the first three numbers as integers
            # state = tuple(int(num_str) -1 for num_str in split_string[:3])
            state = tuple(int(num_str)  for num_str in split_string[:3])
            dict[state] = kwh_per_m2  
            list.append(kwh_per_m2)
            list2.append(state)  
        list3 = [list,list2]
        return list3,dict


    def get_transition_probs(self,state):
        # Initialize the transition_probabilities2 array with the appropriate shape
        transition_probabilities2 = np.zeros((self.num_actions, self.num_states, 3))

        for action in range(self.num_actions):
            # Extract the transition probabilities for the given action and state
            probability_array = self.state_transition_model[action][state].toarray()[0]
            # [num_states x (action x (num.states x 3)) ]
            # Assign transition probabilities, next states, and rewards
            transition_probabilities2[action,:, 0] = probability_array
            transition_probabilities2[action,:, 1] = np.arange(self.num_states)
            transition_probabilities2[action,:, 2] = self.rewards[action, state]
                #     prob = probability_array[next_state]
                #     # reward = self.get_reward(action=action, current_state=current_state)
                #     reward = self.rewards[action,current_state]
                #     transition_probabilities[next_state, :] = [prob, next_state, reward]
                # print("after")
        
        return transition_probabilities2


    def reset(self):
        """
        Resets the environment to its initial state.
        Returns
        -------
        state : int
            The initial state index.
        """
        self.time = 0
        self.current_state = 0
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
        # Get transition probabilities, next state, and reward

        # Update time
        self.time += self.time_step

        # Choose next state based on transition probabilities
        probability_array = self.state_transition_model[action][self.current_state].toarray()[0]
        
        try:
            next_state = np.random.choice(self.num_states, p=probability_array)

        except Exception as e:
            # Handle the specific exception if it occurs
            raise RuntimeError("The command failed: " + str(e) + ":" + str(probability_array))


        
        # Calculate state reward
        
        reward = self.get_reward(action, self.current_state)

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


# if __name__=="__main__":
#     env = House()
#     env.get_transition_probs()
# #     r = env.rewards
# # #     tryout = env.health_age_state_tansition_matrix
# # # #     state= env.num_years
    
# #     probs = env.state_transition_model


    # print('bla')



