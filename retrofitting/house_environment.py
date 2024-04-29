# from gym import Env, spaces
import gymnasium as gym
from gymnasium import Env, spaces
import numpy as np
import pandas as pd
from gamma_deterioration import matrices_gen 
import random
from scipy.sparse import csr_matrix

class House(Env):
    def __init__(self, house_size_m2: float = 120):
        super(House, self).__init__()

        ###################CLASS ATTRIBUTES######################
        self.current_state = 0
        self.time = 0
        self.num_years = 30
        self.time_step = 10
        self.state_space = self.get_state_space(num_damage_states=3,num_years= self.num_years, time_step= self.time_step) 
        self.num_states = len(self.state_space)
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Discrete(self.num_states)
        self.house_size_m2 = house_size_m2
        _, self.kwh_per_state = self.import_simulation_data(file_path='building_scenarios_copy.csv',no_windows=True)
        self.renovation_costs = np.array([0, 3000, 1160, 8000])    # [cost_doNothing, cost_roof, cost_wall, cost_cellar] # TODO: should change according to m2
        # self.energy_demand_nominal = [57, 95, 38]  # [roof, wall, cellar]
        self.degradation_rates = [0.1, 0.2, 0.5]
        self.terminal_state_indices = self.get_terminal_state_indices(self.state_space)
        self.states_probs = self.state_space_probability(state_space = self.state_space,save_version = True , calculate_gamma_distribution_probabilities= False, import_saved_probabilities = False, step_size=self.time_step)
        ###################FUNCTIONS######################

    def one_hot_enc(self,action):
        # Get transition probabilities, next state, and reward
        action_one_hot_enc = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
        act = action_one_hot_enc[action]  
        return act
    
    def import_simulation_data(self,file_path, no_windows: bool):
        data = pd.read_csv(file_path) 
        list = []
        list2 = [] 
        dict = {}
        if no_windows:
            val = 3
        else: 
            val = 1
        for indx in range(0,len(data),val):
            kwh_per_m2 = data['energy[kWh/m2]'][indx]
            string = data['state'][indx]
            # Remove the parentheses and split the string by commas
            split_string = string.strip("()").split(", ")
            # Extract the first three numbers as integers
            state = tuple(int(num_str) - 1 for num_str in split_string[:3])
            dict[state] = kwh_per_m2  
            list.append(kwh_per_m2)
            list2.append(state)  
        list3 = [list,list2]
        return list3,dict
    @staticmethod
    def get_state_space_old(num_damage_states: int, num_years: int, time_step: int):
        state_space = {}
        state_number = 0
        for r_damage_state in range(num_damage_states):
            for w_damage_state in range(num_damage_states):
                for c_damage_state in range(num_damage_states):
                    for age_r in range(0,num_years,time_step):
                        for age_w in range(0,num_years,time_step):
                            for age_f in range(0,num_years,time_step):
                                state_space[state_number] = (r_damage_state, w_damage_state, c_damage_state,age_r,age_w,age_f)
                                state_number += 1
        return state_space 
    @staticmethod
    def get_state_space(num_damage_states: int, num_years: int, time_step: int):
        state_space = {}
        state_number = 0
        for time in range(0,(num_years+time_step),time_step):
            for r_damage_state in range(num_damage_states):
                for w_damage_state in range(num_damage_states):
                    for c_damage_state in range(num_damage_states):
                        for age_r in range(0,num_years+time_step,time_step):
                            for age_w in range(0,num_years+time_step,time_step):
                                for age_f in range(0,num_years+time_step,time_step):
                                    state_space[state_number] = (time,r_damage_state, w_damage_state, c_damage_state,age_r,age_w,age_f)
                                    state_number += 1
        
        # # Create a list to store keys to delete
        # keys_to_delete = []
        # for state in state_space:
        #     if state_space[state][0] < state_space[state][4] or state_space[state][0] < state_space[state][5] or state_space[state][0] < state_space[state][6] :
        #         keys_to_delete.append(state)
        #     elif state_space[state][0] == 0 and state_space[state] != state_space[0]:
        #         keys_to_delete.append(state)

        # # Delete items from the dictionary
        # for key in keys_to_delete:
        #     del state_space[key]

        # new_dict = {}
        # for index, state_name in enumerate(state_space):
        #     new_dict[index] = state_space[state_name]

        return state_space

    def get_terminal_state_indices(self,state_space): 
        list = []
        for index,current_state in state_space.items(): 
            if current_state[3] == self.num_years or current_state[4] == self.num_years or current_state[5] ==self.num_years: 
                list.append(index)
        return list
    @staticmethod
    def energy2euros(num_of_kwh: float) -> float:
        price_kwh = 0.35
        return price_kwh * num_of_kwh

    def get_reward(self, action: int, current_state: int) -> float:
        action_costs = self.renovation_costs[action]
        state_name = self.state_space[current_state][1:4]
        total_energy_demand = self.kwh_per_state[state_name]
        if total_energy_demand > 210:
            total_energy_demand = total_energy_demand * 2
        energy_bills = House.energy2euros(total_energy_demand)
        energy_bills = energy_bills * self.house_size_m2
        
        if self.state_space[current_state][0] == 0 :
            net_cost = action_costs + energy_bills
        elif self.state_space[current_state][0] == self.num_years:
            net_cost = 0 
        else :
            net_cost = action_costs + (energy_bills*self.time_step)
        reward = -net_cost
        return reward
    
    def get_reward_old(self, action: int, current_state: int) -> float:
        action_costs = self.renovation_costs[action]
        state_name = self.state_space[current_state][0:3]
        total_energy_demand = self.kwh_per_state[state_name]
        energy_bills = House.energy2euros(total_energy_demand)
        energy_bills = energy_bills * self.house_size_m2
        net_cost = action_costs + energy_bills
        reward = -net_cost
        return reward
    #complex_wrong but works
    def state_space_probability0(self,state_space: dict,save_version:bool , calculate_gamma_distribution_probabilities:bool, import_saved_probabilities: bool, step_size:int):
            if import_saved_probabilities :
                array_of_arrays = np.load('array_of_arrays.npy')
            else:
            
                if calculate_gamma_distribution_probabilities == False:
                    load = np.load("transition_matrices.npy")
                else : 
                    load = matrices_gen(SIMPLE_STUFF = True, N = 1000000, T = self.num_years, do_plot = False)
                
                probability_matrices = load 
                action_matrices =np.array([[1., 0., 0.], [1., 0., 0.],[1., 0., 0.]])


                num_states = len(state_space)
                probability_array = np.zeros((4, num_states, num_states), dtype=np.float32)


                # Compute probabilities using vectorized operations
                for action in range(self.action_space.n):
                    overall_probability = probability_array[action]
                    act = self.one_hot_enc(action)
                    for i in range(num_states):
                        for future in range(num_states):
                            if act[0] == 0: 
                                #if the next age of the roof is bigger by one time step than the current age, get probability
                                #othewise probability = 0
                                #This ensures that the only possible states that can be reached from the state that we are in, are the next age-states
                                if state_space[future][3] == (state_space[i][3]+ step_size) : 
                                    prob1 = probability_matrices[state_space[i][3], state_space[i][0], state_space[future][0]] 
                                else:  
                                    prob1 = 0
                            else: 
                                #if the action for this component is not 0 , meaning that the roof has to be changed 
                                #consider only the states with age 0 as an option 
                                if state_space[future][3] == (state_space[i][3]+ step_size) : 
                                    prob1 = action_matrices[state_space[i][0],state_space[future][0]]
                                else:
                                    prob1 = 0

                            if act[1] == 0:
                                #if the next age of the wall is bigger by one time step than the current age, get probability
                                #othewise probability = 0
                                #This ensures that the only possible states that can be reached from the state that we are in, are the next age-states
                                if state_space[future][4] == (state_space[i][4]+ step_size) :
                                    prob2 = probability_matrices[state_space[i][4],state_space[i][1],state_space[future][1]] 
                                else :
                                    prob2 = 0
                            else: 
                                #if the action for this component is not 0 , meaning that the roof has to be changed 
                                #consider only the states with age 0 as an option 
                                if state_space[future][4] == (state_space[i][4]+ step_size) :
                                    prob2 = action_matrices[state_space[i][1],state_space[future][1]]
                                else:
                                    prob2 = 0

                            if act[2] == 0: 
                                #same here for the floor
                                if state_space[future][5] == (state_space[i][5]+ step_size):
                                    prob3 = probability_matrices[state_space[i][5],state_space[i][2],state_space[future][2]] 
                                else:
                                    prob3 = 0
                            else:
                                #same here as above
                                if state_space[future][5] == (state_space[i][5]+step_size):
                                    prob3 = action_matrices[state_space[i][2],state_space[future][2]] 
                                else: 
                                    prob3 = 0   
                            
                            overall_probability[i,future] = prob1 * prob2 * prob3
                    print(action)
                    # array.append(overall_probability)
                    


                bla = probability_array[0]

                def normalize_probabilities(arrays):
                    normalized_arrays = []
                    for array in arrays:
                        row_sums = [sum(row) for row in array]
                        normalized_array = []
                        for i, row in enumerate(array):
                            if row_sums[i] != 0:
                                normalized_row = [value / row_sums[i] for value in row]
                            else:
                                normalized_row = [0] * len(row)  # Handle division by zero
                            normalized_array.append(normalized_row)
                        normalized_arrays.append(normalized_array)
                    return normalized_arrays

                # Assuming arr is your array of arrays
                row_sums_lists = []

                # Compute the sums of each row for each array
                for array in range(len(probability_array)):
                    array_row_sums = []
                    for row in probability_array[array]:
                        row_sum = np.sum(row)
                        array_row_sums.append(row_sum)
                    row_sums_lists.append(array_row_sums)


                # Normalize arrays
                normalized_arrays = normalize_probabilities(probability_array)

                # Assuming arr is your array of arrays
                row_sums_lists2 = []

                # Compute the sums of each row for each array
                for array in normalized_arrays:
                    array_row_sums = []
                    for row in array:
                        row_sum = np.sum(row)
                        array_row_sums.append(row_sum)
                    row_sums_lists2.append(array_row_sums)

                #conbine the four arrays into one array of arrays
                array_of_arrays = np.array(normalized_arrays)

                if save_version:
                    np.save("array_of_arrays.npy", array_of_arrays)
        
            return array_of_arrays
    #complex right but doesn't work
    def state_space_probability0(self,state_space: dict,save_version:bool , calculate_gamma_distribution_probabilities:bool, import_saved_probabilities: bool, step_size:int):
        if import_saved_probabilities :
            array_of_arrays = np.load('array_of_arrays.npy')
        else:
        
            if calculate_gamma_distribution_probabilities == False:
                load = np.load("transition_matrices.npy")
            else : 
                load = matrices_gen(SIMPLE_STUFF = True, N = 1000000, T = self.num_years, do_plot = False)
            
            probability_matrices = load 
            action_matrices =np.array([[1., 0., 0.], [1., 0., 0.],[1., 0., 0.]])


            num_states = len(state_space)
            probability_array = np.zeros((4, num_states, num_states), dtype=np.float32)


            # Compute probabilities using vectorized operations
            for action in range(self.action_space.n):
                overall_probability = probability_array[action]
                act = self.one_hot_enc(action)
                for i in range(num_states):
                    for future in range(num_states):
                        if act[0] == 0: 
                            #if the next age of the roof is bigger by one time step than the current age, get probability
                            #othewise probability = 0
                            #This ensures that the only possible states that can be reached from the state that we are in, are the next age-states
                            if state_space[future][3] == (state_space[i][3]+ step_size) : 
                                prob1 = probability_matrices[state_space[i][3], state_space[i][0], state_space[future][0]] 
                            else:  
                                prob1 = 0
                        else: 
                            #if the action for this component is not 0 , meaning that the roof has to be changed 
                            #consider only the states with age 0 as an option 
                            if state_space[future][3] == (state_space[0][3]) : 
                                prob1 = action_matrices[state_space[i][0],state_space[future][0]]
                            else:
                                prob1 = 0

                        if act[1] == 0:
                            #if the next age of the wall is bigger by one time step than the current age, get probability
                            #othewise probability = 0
                            #This ensures that the only possible states that can be reached from the state that we are in, are the next age-states
                            if state_space[future][4] == (state_space[i][4]+ step_size) :
                                prob2 = probability_matrices[state_space[i][4],state_space[i][1],state_space[future][1]] 
                            else :
                                prob2 = 0
                        else: 
                            #if the action for this component is not 0 , meaning that the roof has to be changed 
                            #consider only the states with age 0 as an option 
                            if state_space[future][4] == (state_space[0][4]) :
                                prob2 = action_matrices[state_space[i][1],state_space[future][1]]
                            else:
                                prob2 = 0

                        if act[2] == 0: 
                            #same here for the floor
                            if state_space[future][5] == (state_space[i][5]+ step_size):
                                prob3 = probability_matrices[state_space[i][5],state_space[i][2],state_space[future][2]] 
                            else:
                                prob3 = 0
                        else:
                            #same here as above
                            if state_space[future][5] == (state_space[0][5]):
                                prob3 = action_matrices[state_space[i][2],state_space[future][2]] 
                            else: 
                                prob3 = 0   
                        
                        overall_probability[i,future] = prob1 * prob2 * prob3
                print(action)
                # array.append(overall_probability)
                


            bla = probability_array[0]

            def normalize_probabilities(arrays):
                normalized_arrays = []
                for array in arrays:
                    row_sums = [sum(row) for row in array]
                    normalized_array = []
                    for i, row in enumerate(array):
                        if row_sums[i] != 0:
                            normalized_row = [value / row_sums[i] for value in row]
                        else:
                            normalized_row = [0] * len(row)  # Handle division by zero
                        normalized_array.append(normalized_row)
                    normalized_arrays.append(normalized_array)
                return normalized_arrays

            # Assuming arr is your array of arrays
            row_sums_lists = []

            # Compute the sums of each row for each array
            for array in range(len(probability_array)):
                array_row_sums = []
                for row in probability_array[array]:
                    row_sum = np.sum(row)
                    array_row_sums.append(row_sum)
                row_sums_lists.append(array_row_sums)


            # Normalize arrays
            normalized_arrays = normalize_probabilities(probability_array)

            # Assuming arr is your array of arrays
            row_sums_lists2 = []

            # Compute the sums of each row for each array
            for array in normalized_arrays:
                array_row_sums = []
                for row in array:
                    row_sum = np.sum(row)
                    array_row_sums.append(row_sum)
                row_sums_lists2.append(array_row_sums)

            #conbine the four arrays into one array of arrays
            array_of_arrays = np.array(normalized_arrays)

            if save_version:
                np.save("array_of_arrays.npy", array_of_arrays)
    
        return array_of_arrays
    #complex right adjusted to new state space
    def state_space_probability(self,state_space: dict,save_version:bool , calculate_gamma_distribution_probabilities:bool, import_saved_probabilities: bool, step_size:int):
        if import_saved_probabilities :
            array_of_arrays = np.load('array_of_arrays.npy')
        else:
        
            if calculate_gamma_distribution_probabilities == False:
                load = np.load("transition_matrices.npy")
            else : 
                load = matrices_gen(SIMPLE_STUFF = True, N = 1000000, T = self.num_years, do_plot = False)
            
            probability_matrices = load 
            action_matrices =np.array([[1., 0., 0.], [1., 0., 0.],[1., 0., 0.]])

            # initial_probabilities = np.array(pro)    
            num_states = len(state_space)
            make_zero_array = np.zeros((4, num_states, num_states), dtype=np.float32)
            # Convert the dense array to a sparse matrix
            probability_array = csr_matrix(make_zero_array)


            # Compute probabilities using vectorized operations
            for action in range(self.action_space.n):
                overall_probability = probability_array[action]
                act = self.one_hot_enc(action)
                for count, i in enumerate(state_space.keys()):
                    for f_count, future in enumerate(state_space.keys()):
                        if state_space[0] != self.num_years: 
                            if act[0] == 0: 
                                #if the next age of the roof is bigger by one time step than the current age, get probability
                                #othewise probability = 0
                                #This ensures that the only possible states that can be reached from the state that we are in, are the next age-states
                                if state_space[future][4] == (state_space[i][4]+ step_size) and state_space[future][0] == (state_space[i][0]+ step_size): 
                                    prob1 = probability_matrices[state_space[i][4], state_space[i][1], state_space[future][1]] 
                                else:  
                                    prob1 = 0
                            else: 
                                #if the action for this component is not 0 , meaning that the roof has to be changed 
                                #consider only the states with age 0 as an option 
                                if state_space[future][4] == (state_space[0][4]) and state_space[future][0] == (state_space[i][0]+ step_size): 
                                    prob1 = action_matrices[state_space[i][1],state_space[future][1]]
                                else:
                                    prob1 = 0

                            if act[1] == 0:
                                #if the next age of the wall is bigger by one time step than the current age, get probability
                                #othewise probability = 0
                                #This ensures that the only possible states that can be reached from the state that we are in, are the next age-states
                                if state_space[future][5] == (state_space[i][5]+ step_size) and state_space[future][0] == (state_space[i][0]+ step_size) :
                                    prob2 = probability_matrices[state_space[i][5],state_space[i][2],state_space[future][2]] 
                                else :
                                    prob2 = 0
                            else: 
                                #if the action for this component is not 0 , meaning that the roof has to be changed 
                                #consider only the states with age 0 as an option 
                                if state_space[future][5] == (state_space[0][5]) and state_space[future][0] == (state_space[i][0]+ step_size) :
                                    prob2 = action_matrices[state_space[i][2],state_space[future][2]]
                                else:
                                    prob2 = 0

                            if act[2] == 0: 
                                #same here for the floor
                                if state_space[future][6] == (state_space[i][6]+ step_size) and state_space[future][0] == (state_space[i][0]+ step_size):
                                    prob3 = probability_matrices[state_space[i][6],state_space[i][3],state_space[future][3]] 
                                else:
                                    prob3 = 0
                            else:
                                #same here as above
                                if state_space[future][6] == (state_space[0][6]) and state_space[future][0] == (state_space[i][0]+ step_size):
                                    prob3 = action_matrices[state_space[i][3],state_space[future][3]] 
                                else: 
                                    prob3 = 0   
                            
                            overall_probability[count,f_count] = prob1 * prob2 * prob3

                        elif state_space[i] == state_space[self.num_years] :
                            overall_probability[count,f_count] = 1   
                        else : 
                            overall_probability[count,f_count] = 0

                print(action)
                # array.append(overall_probability)
                


            bla = probability_array[0]

            def normalize_probabilities(arrays):
                normalized_arrays = []
                for array in arrays:
                    row_sums = [sum(row) for row in array]
                    normalized_array = []
                    for i, row in enumerate(array):
                        if row_sums[i] != 0:
                            normalized_row = [value / row_sums[i] for value in row]
                        else:
                            normalized_row = [0] * len(row)  # Handle division by zero
                        normalized_array.append(normalized_row)
                    normalized_arrays.append(normalized_array)
                return normalized_arrays

            # Assuming arr is your array of arrays
            row_sums_lists = []

            # Compute the sums of each row for each array
            for array in range(len(probability_array)):
                array_row_sums = []
                for row in probability_array[array]:
                    row_sum = np.sum(row)
                    array_row_sums.append(row_sum)
                row_sums_lists.append(array_row_sums)


            # Normalize arrays
            normalized_arrays = normalize_probabilities(probability_array)

            # Assuming arr is your array of arrays
            row_sums_lists2 = []

            # Compute the sums of each row for each array
            for array in normalized_arrays:
                array_row_sums = []
                for row in array:
                    row_sum = np.sum(row)
                    array_row_sums.append(row_sum)
                row_sums_lists2.append(array_row_sums)

            #conbine the four arrays into one array of arrays
            array_of_arrays = np.array(normalized_arrays)

            if save_version:
                np.save("array_of_arrays.npy", array_of_arrays)
    
        return array_of_arrays
    #simple right
    def state_space_probability0(self,state_space: dict,save_version:bool , calculate_gamma_distribution_probabilities:bool, import_saved_probabilities: bool, step_size:int):
        
        probabilities = [[0.2,0.8,0],[0,0.4,0.6],[0,0,1]],[[1,0,0],[1,0,0],[1,0,0]]   
        probability_matrices = probabilities
        num_states = len(state_space)
        probability_array = np.zeros((4, num_states, num_states), dtype=np.float32)


        # Compute probabilities using vectorized operations
        for action in range(self.action_space.n):
            overall_probability = probability_array[action]
            act = self.one_hot_enc(action)
            for i in range(num_states):
                for future in range(num_states):
                    if act[0] == 0: 
                        #if the next age of the roof is bigger by one time step than the current age, get probability
                        #othewise probability = 0
                        #This ensures that the only possible states that can be reached from the state that we are in, are the next age-states
                        if state_space[future][3] == (state_space[i][3]+ step_size) : 
                            prob1 = probability_matrices[0][state_space[i][0]][state_space[future][0]] 
                        else:  
                            prob1 = 0
                    else: 
                        #if the action for this component is not 0 , meaning that the roof has to be changed 
                        #consider only the states with age 0 as an option 
                        if state_space[future][3] == (state_space[0][3]) : 
                            prob1 = probability_matrices[1][state_space[i][0]][state_space[future][0]] 
                        else:
                            prob1 = 0


                    if act[1] == 0:
                        #if the next age of the wall is bigger by one time step than the current age, get probability
                        #othewise probability = 0
                        #This ensures that the only possible states that can be reached from the state that we are in, are the next age-states
                        if state_space[future][4] == (state_space[i][4]+ step_size) :
                            prob2 = probability_matrices[0][state_space[i][1]][state_space[future][1]]
                        else :
                            prob2 = 0
                    else: 
                        #if the action for this component is not 0 , meaning that the roof has to be changed 
                        #consider only the states with age 0 as an option 
                        if state_space[future][4] == (state_space[0][4]) :
                            prob2 = probability_matrices[1][state_space[i][1]][state_space[future][1]]
                        else:
                            prob2 = 0

                    if act[2] == 0: 
                        #same here for the floor
                        if state_space[future][5] == (state_space[i][5]+ step_size):
                            prob3 = probability_matrices[0][state_space[i][2]][state_space[future][2]] 
                        else:
                            prob3 = 0
                    else:
                        #same here as above
                        if state_space[future][5] == (state_space[0][5]):
                            prob3 = probability_matrices[1][state_space[i][2]][state_space[future][2]] 
                        else: 
                            prob3 = 0   
                    
                    overall_probability[i,future] = prob1 * prob2 * prob3
            print(action)
            # array.append(overall_probability)
        

        def normalize_probabilities(arrays):
            normalized_arrays = []
            for array in arrays:
                row_sums = [sum(row) for row in array]
                normalized_array = []
                for i, row in enumerate(array):
                    if row_sums[i] != 0:
                        normalized_row = [value / row_sums[i] for value in row]
                    else:
                        normalized_row = [0] * len(row)  # Handle division by zero
                    normalized_array.append(normalized_row)
                normalized_arrays.append(normalized_array)
            return normalized_arrays

        # Assuming arr is your array of arrays
        row_sums_lists = []

        # Compute the sums of each row for each array
        for array in range(len(probability_array)):
            array_row_sums = []
            for row in probability_array[array]:
                row_sum = np.sum(row)
                array_row_sums.append(row_sum)
            row_sums_lists.append(array_row_sums)


        # Normalize arrays
        normalized_arrays = normalize_probabilities(probability_array)

        # Assuming arr is your array of arrays
        row_sums_lists2 = []

        # Compute the sums of each row for each array
        for array in normalized_arrays:
            array_row_sums = []
            for row in array:
                row_sum = np.sum(row)
                array_row_sums.append(row_sum)
            row_sums_lists2.append(array_row_sums)

        #conbine the four arrays into one array of arrays
        array_of_arrays = np.array(normalized_arrays)

        if save_version:
            np.save("array_of_arrays.npy", array_of_arrays)

        return array_of_arrays

    '''def state_space_probability(self,state_space: dict,save_version:bool , calculate_gamma_distribution_probabilities:bool, import_saved_probabilities: bool, step_size:int):
            if import_saved_probabilities :
                array_of_arrays = np.load('array_of_arrays.npy')
            else:
            
                if calculate_gamma_distribution_probabilities == False:
                    load = np.load("transition_matrices.npy")
                else : 
                    load = matrices_gen(SIMPLE_STUFF = True, N = 1000000, T = self.num_years, do_plot = False)
                
                probability_matrices = load 
                action_matrices =np.array([[1., 0., 0.], [1., 0., 0.],[1., 0., 0.]])
                num_states = len(state_space)
                probability_array = np.zeros((4, num_states, num_states), dtype=np.float32)
                dictionary  = {}    

                # Compute probabilities using vectorized operations
                for action in range(self.action_space.n):
                    overall_probability = probability_array[action]
                    act = self.one_hot_enc(action)
                    for i in range(num_states):
                        for future in range(num_states):
                            if act[0] == 0: # if the action for this component equals to 
                                if state_space[future][3] == (state_space[i][3]+ step_size) : #if the next age of the roof is bigger by one time step than the current age, get probability
                                    prob1 = probability_matrices[state_space[i][3], state_space[i][0], state_space[future][0]] 
                                else: #otherwise probability is 0
                                    prob1 = 0
                            else: 
                                if state_space[future][3] == 0 : 
                                    prob1 = action_matrices[state_space[i][0],state_space[future][0]]
                                else:
                                    prob1 = 0

                            if act[1] == 0:
                                if state_space[future][4] == (state_space[i][4]+ step_size) :
                                    prob2 = probability_matrices[state_space[i][4],state_space[i][1],state_space[future][1]] 
                                else :
                                    prob2 = 0
                            else: 
                                if state_space[future][4] == 0 :
                                    prob2 = action_matrices[state_space[i][1],state_space[future][1]]
                                else:
                                    prob2 = 0

                            if act[2] == 0: 
                                if state_space[future][5] == (state_space[i][5]+ step_size):
                                    prob3 = probability_matrices[state_space[i][5],state_space[i][2],state_space[future][2]] 
                                else:
                                    prob3 = 0
                            else:
                                if state_space[future][5] == 0:
                                    prob3 = action_matrices[state_space[i][2],state_space[future][2]] 
                                else: 
                                    prob3 = 0   
                            
                            probability = prob1 * prob2 * prob3
                            overall_probability[i,future] = prob1 * prob2 * prob3
                            # if probability != 0: 
                            #     dictionary[i][future] = probability
                        
                    print(action)
                    # array.append(overall_probability)

                # def normalize_probabilities(dictionaries):
                #     dictionaries = dict
                #     for dictionary in range(len(dictionaries)):
                #         total_sum = sum(dictionary.values())
                #         for key in dictionary: 
                #             dictionary[key] /= total_sum
                    return  overall_probability

            # Assuming arr is your array of arrays
            row_sums_lists = []

            # Compute the sums of each row for each array
            for array in range(len(probability_array)):
                array_row_sums = []
                for row in probability_array[array]:
                    row_sum = np.sum(row)
                    array_row_sums.append(row_sum)
                row_sums_lists.append(array_row_sums)


            # Normalize arrays
            normalized_arrays = normalize_probabilities(probability_array)

            # Assuming arr is your array of arrays
            row_sums_lists2 = []

            # Compute the sums of each row for each array
            for array in normalized_arrays:
                array_row_sums = []
                for row in array:
                    row_sum = np.sum(row)
                    array_row_sums.append(row_sum)
                row_sums_lists2.append(array_row_sums)

            #conbine the four arrays into one array of arrays
            array_of_arrays = np.array(normalized_arrays)

            if save_version:
                np.save("array_of_arrays.npy", array_of_arrays)
        
            return array_of_arrays'''

    def get_transition_probs_old(self, current_state: int, action: int):
        transition_probabilities = []
        for next_state in range(self.num_states+1):
            prob = self.states_probs[action][current_state][next_state]
            reward = self.get_reward(action=action, current_state=current_state)
            transition_probabilities.append((prob, next_state, reward))
        return transition_probabilities
    
    def get_transition_probs(self, current_state: int, action: int):
        transition_probabilities = []
        for next_state in range(self.num_states):
            prob = self.states_probs[action][current_state][next_state]
            reward = self.get_reward(action=action, current_state=current_state)
            transition_probabilities.append((prob, next_state, reward))
        return transition_probabilities

    def reset(self, seed = random):
        self.time = 0
        self.current_state = 0
        #reset should have observation
        return 0  # Return initial state index
    
    def step(self, action):
        # Get all transition probabilities for the current state
        transitions = self.states_probs[action][self.current_state]
        if np.sum(transitions) == 0: 
            done = True
            next_state = self.num_states - 1
            self.time += self.time_step
        else: 
            # Extract probabilities from transition probabilities
            # probabilities = [prob for prob in transitions]
            # Choose next state based on transition probabilities
            next_state = np.random.choice(self.num_states, p=transitions)
            self.time += self.time_step
            # Check if episode is done (time limit reached)
            done = self.time >= self.num_years or self.current_state == self.num_states
        # Calculate state reward
        reward = self.get_reward(action, self.current_state)   
        # Update time
        self.current_state = next_state
        return next_state, reward, done, {}

    def render(self, mode='human'):
        # Optionally render the environment
        pass

    def close(self):
        # Clean up resources, if any
        pass




if __name__=="__main__":
    env = House()
    n = np.load('array_of_arrays.npy')
    bla = env.get_transition_probs(current_state =2674 , action = 0)
    # bla = env.get_reward(action=0, current_state =0)
    states = env.state_space
    s = states[25]
    print(s)
    print(bla)
    states = env.state_space
# #     not_terminal_states = env.not_terminal_states
    try1 = env.states_probs 

    # np.save('probabilities-try.npy',try1)
    see = np.load('probabilities-try.npy')
    print('bla')
   