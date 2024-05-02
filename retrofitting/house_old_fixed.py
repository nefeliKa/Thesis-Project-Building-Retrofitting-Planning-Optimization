from gym import Env, spaces
import numpy as np
import pandas as pd
import pickle 
import csv
from scipy.sparse import lil_matrix, save_npz
from gamma_deterioration import matrices_gen 
import time
from scipy.sparse import csr_matrix
from multiprocessing import Pool





# Define a function to process actions in parallel
def process_action(action, state_space, time_step, num_years, num_damage_states, action_one_hot_enc, health_age_state_tansition_matrix, health_age_state_space):
    sparse_matrix = lil_matrix((len(state_space), len(state_space)), dtype=np.float64)
    print(np.shape(state_space))
    for current_state_index, current_state in enumerate(state_space):
        if current_state[0] == num_years:
            sparse_matrix[current_state_index, current_state_index] = 1
            continue
        if action == 0:
            future_states_indices = np.where((state_space[:, 0] == current_state[0] + time_step) &
                                            (state_space[:, 4] > current_state[4]) &
                                            (state_space[:, 5] > current_state[5]) &
                                            (state_space[:, 6] > current_state[6]))[0]
        elif action == 1:
            future_states_indices = np.where((state_space[:, 0] == current_state[0] + time_step) & 
                                            (state_space[:, 4] == 0) &
                                            (state_space[:, 5] > current_state[5]) &
                                            (state_space[:, 6] > current_state[6]))[0]
        elif action == 2:
            future_states_indices = np.where((state_space[:, 0] == current_state[0] + time_step) &
                                            (state_space[:, 5] == 0) &
                                            (state_space[:, 4] > current_state[4]) &
                                            (state_space[:, 6] > current_state[6]))[0]
        else:
            future_states_indices = np.where((state_space[:, 0] == current_state[0] + time_step) &
                                            (state_space[:, 4] > current_state[4]) &
                                            (state_space[:, 5] > current_state[5]) &
                                            (state_space[:, 6] == 0))[0]

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
                sparse_matrix[current_state_index, future_state_index] = new_probability

    return sparse_matrix


class House(Env):
    def __init__(self, house_size_m2: float = 120):
        super(House, self).__init__()
        self.num_damage_states = 3
        self.num_actions = 4
        ##### ACTIONS #####
        # 0,  # DO_NOTHING
        # 1,  # FIX_ROOF
        # 2,  # FIX_WALL
        # 3   # FIX_FACADE
        self.current_state = 0
        self.time = 0
        self.num_years = 60
        self.time_step = 5
        self.action_space = spaces.Discrete(4)
        self.state_space = House.get_state_space(num_damage_states=self.num_damage_states,
                                                 num_years= self.num_years,
                                                 time_step = self.time_step)
        self.num_states = len(self.state_space)
        self.observation_space = spaces.Discrete(self.num_states)

        _, self.kwh_per_state = self.import_simulation_data(file_path='building_scenarios_copy.csv',no_windows=True)
        self.house_size_m2 = house_size_m2

        # # [cost_doNothing, cost_roof, cost_wall, cost_cellar]
        self.renovation_costs = np.array([0, 13791, 16273, 5000])  # 

        # # [roof, wall, cellar]
        # # self.energy_demand_nominal = [57, 95, 38]
        self.degradation_rates = [0.0, 0.2, 0.5]
        self.energy_bills = self.get_state_electricity_bill(state_space = self.state_space,kwh_per_state=self.kwh_per_state)
        self.material_probability_matrices,self.action_matrices = \
                                                self.import_gamma_probabilities(calculate_gamma_distribution_probabilities= True,
                                                step_size=self.time_step,SIMPLE_STUFF = True, 
                                                N = 1000, do_plot= False, T = self.num_years+self.time_step,
                                                save_probabilities = True)
        self.health_age_state_space, self.health_age_state_tansition_matrix = self.health_age_state_tansitions(num_damage_states= self.num_damage_states,num_years = self.num_years,material_probability_matrices = self.material_probability_matrices, action_matrices=self.action_matrices,time_step = self.time_step)        # self.state_transition_model = House.get_state_transition_model(num_actions=self.num_actions, state_space=self.state_space, time_step= self.time_step,num_years = self.num_years)
        # self.state_transition_model = self.get_state_transition_model(num_actions=self.num_actions,
        #                                                               state_space=self.state_space, 
        #                                                               time_step= self.time_step,
        #                                                               num_years = self.num_years,
        #                                                               material_probability_matrices= self.material_probability_matrices,
        #                                                               action_matrices=self.action_matrices)
        self.state_transition_model = self.get_state_transition_model(num_actions=self.num_actions,
                                                                      state_space=self.state_space, 
                                                                      time_step= self.time_step,
                                                                      num_years = self.num_years,
                                                                      material_probability_matrices= self.material_probability_matrices,
                                                                      action_matrices=self.action_matrices,
                                                                      health_age_state_tansition_matrix =self.health_age_state_tansition_matrix,
                                                                      health_age_state_space= self.health_age_state_space)
 
####################################################################################################


 
#State space as a numpy with ages and time
    def get_state_space_old(num_damage_states: int, num_years: int, time_step: int):
            state_space = np.empty((0,7))
            # state_number = 0
            for time in range(0,(num_years+time_step),time_step):
                for r_damage_state in range(num_damage_states):
                    for w_damage_state in range(num_damage_states):
                        for c_damage_state in range(num_damage_states):
                            for age_r in range(0,num_years+time_step,time_step):
                                for age_w in range(0,num_years+time_step,time_step):
                                    for age_f in range(0,num_years+time_step,time_step):
                                        state = np.array([time,r_damage_state,w_damage_state,c_damage_state,age_r,age_w,age_f])
                                        state_space = np.vstack([state_space, state])
                                        # state_space[state_number] = (time,r_damage_state, w_damage_state, c_damage_state,age_r,age_w,age_f)
                                        # state_number += 1

            return state_space

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

#State space as a numpy with only ages for the transition models.
    def get_state_space_np_ages(num_damage_states: int, num_years: int, time_step: int):
                state_space = np.empty((0,6))
                # state_number = 0
                for r_damage_state in range(num_damage_states):
                    for w_damage_state in range(num_damage_states):
                        for c_damage_state in range(num_damage_states):
                            for age_r in range(0,num_years+time_step,time_step):
                                for age_w in range(0,num_years+time_step,time_step):
                                    for age_f in range(0,num_years+time_step,time_step):
                                        state = np.array([r_damage_state,w_damage_state,c_damage_state,age_r,age_w,age_f])
                                        state_space = np.vstack([state_space, state])
                                        # state_space[state_number] = (time,r_damage_state, w_damage_state, c_damage_state,age_r,age_w,age_f)
                                        # state_number += 1
                return state_space
    
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

#newer, using np. still slow
    def get_state_transition_model_newer(self, num_actions: int, state_space: dict, time_step:int, num_years:int,material_probability_matrices:np.array, action_matrices:np.array):
                import_probs = False
                list_probs = np.load('probabilities_np.npy')
                if import_probs != True:
                    num_damage_states = 3  # good, medium, bad

                    # Calculate transition model of system
                    action_one_hot_enc = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
                    list_probs = []
                    state_space_length = len(state_space)
                    prob_matrix = np.zeros((4, state_space_length, state_space_length))
                    # number = 0

                    for action in range(num_actions): 
                        # number+=1
                        number = 0
                        # action = 2
                        for current_state_index,current_state in enumerate(state_space): # get state number
                            for future_state_index,future_state in enumerate(state_space): # get future state index number  # get the tuple describing the next state 
                                if future_state[0] == current_state[0] + time_step:
                                    future_states_probabilities = np.zeros(3)
                                    if int(current_state[0]) != num_years:
                                        for i in range(num_damage_states): #get the damage state 
                                            action_array = action_one_hot_enc[action][i] #get the action for that
                                            current_component_st = int(current_state[i+1]) # since the first number is the time, start with the second
                                            future_component_st = int(future_state[i+1])
                                            current_component_age = int(current_state[i+4]) 
                                            future_component_age = int(future_state[i+4])
                                            age_to_index_convertion = int(current_component_age/time_step) if current_component_age != 0 else 0
                                            if action_array == 0 :
                                                if future_component_age == current_component_age+time_step :
                                                    probability = material_probability_matrices[age_to_index_convertion][current_component_st][future_component_st]
                                                else :
                                                    probability = 0
                                            elif   future_component_age == 0: 
                                                probability = action_matrices[current_component_st][future_component_st]
                                            else: 
                                                probability = 0
                                            future_states_probabilities[i] = probability
                                        
                                        new_probability = np.prod(future_states_probabilities) 
                                    else: 
                                        new_probability = 0 
                                    if new_probability != 0:
                                        prob_matrix[action][current_state_index][future_state_index] = new_probability
                                    # toc = time.time()
                                    # print(toc - tic)
                                elif current_state[0] == num_years  and tuple(current_state) == tuple(future_state):
                                    new_probability = 1 # make sure that the final state can only go to itself and no other state 
                                    prob_matrix[action][current_state_index][future_state_index] = new_probability

                                else:  
                                    new_probability = 0 # make sure that the final state can only go to itself and no other state 
                                    prob_matrix[action][current_state_index][future_state_index] = new_probability
                        
                        print(number)
                    

                    copy = prob_matrix
                    # Iterate over each 2D array in the 3D array
                    for array_ind, array_2d in enumerate(prob_matrix):
                        # Iterate over each row in the 2D array
                        for row_ind, row in enumerate(array_2d):
                            # Calculate the sum of the row
                            row_sum = np.sum(row)
                            # Divide each element of the row by the sum of the row
                            if row_sum != 0 : 
                                row_normalized = row / row_sum 
                                copy[array_ind][row_ind] = row_normalized
                            


                    # Iterate over each 2D array in the 3D array
                    for idx, array_2d in enumerate(copy):
                        # Calculate the sum of each row in the 2D array
                        row_sums = np.sum(array_2d, axis=1)
                        # Find the indices of rows where the sum is different from 0 or 1
                        rows_not_zero_or_one = np.where((row_sums != 0.0) & (row_sums != 1.0))[0]
                        # Print the indices of rows that do not sum up to 0 or 1
                        if len(rows_not_zero_or_one) > 0:
                            print(f"In array {idx}, the following rows do not sum up to 0 or 1:")
                            print(rows_not_zero_or_one)
                        else:
                            print(f"In array {idx}, all rows sum up to 0 or 1.")

                    # print('bla')

                    # concatenated_array = np.array(list_probs)
                    np.save('probabilities_np.npy',copy)
                return prob_matrix

#newer, using np., a bit faster,wth sparse matrices 
    def get_state_transition_model_old(self, num_actions: int, state_space: dict, time_step:int, num_years:int,material_probability_matrices:np.array, action_matrices:np.array):
                    num_damage_states = 3  # good, medium, bad

                    # Calculate transition model of system
                    action_one_hot_enc = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
                    list_probs = []
                    # number = 0

                    for action in range(num_actions): 
                        # number+=1
                        arrays = []
                        number = 0
                        # Get all rows of state_space where the first value is 0
                        states_t0 = state_space[state_space[:, 0] == 0]
                        states_t1 = state_space[state_space[:, 0] == time_step]
                        # assert len(states_t0) == len(states_t1)
                        transitions_t0_t1_1 = lil_matrix((len(states_t0), len(states_t1)))
                        transitions_t0_t1 = np.zeros((len(states_t0), len(states_t1)))
                        row_transitions_t0_t1 = 0
                        for current_state in states_t0: # get state number
                            # tic = time.time()
                            col_transitions_t0_t1 = 0
                            for future_state in states_t1: # get future state index number
                                future_states_probabilities = np.zeros(3)   # 1 prob per component (Wall, Roof, Cellar)
                                for i in range(num_damage_states): #get the damage state 
                                    component_action = action_one_hot_enc[action][i] #get the action for that (do nothing OR renovate)
                                    current_component_st = int(current_state[i+1]) # since the first number is the time, start with the second
                                    future_component_st = int(future_state[i+1])
                                    current_component_age = int(current_state[i+4]) 
                                    future_component_age = int(future_state[i+4])
                                    age_to_index_convertion = int(current_component_age/time_step) if current_component_age != 0 else 0
                                    if component_action == 0 :  # Do nothing
                                        if future_component_age == current_component_age+time_step:
                                            probability = material_probability_matrices[age_to_index_convertion][current_component_st][future_component_st]
                                        else :
                                            probability = 0
                                    else:                       # Renovate
                                        probability = action_matrices[current_component_st][future_component_st]
                                    future_states_probabilities[i] = probability
                                
                                total_prob = np.prod(future_states_probabilities)   
                                transitions_t0_t1_1[row_transitions_t0_t1, col_transitions_t0_t1] = total_prob
                                transitions_t0_t1[row_transitions_t0_t1, col_transitions_t0_t1] = total_prob
                                col_transitions_t0_t1 += 1
                            # assert np.isclose(np.sum(transitions_t0_t1[row_transitions_t0_t1, :]), 1, atol=1e-5)

                            row_transitions_t0_t1 += 1
                                # toc = time.time()
                                # print(toc - tic)

                            # array = np.vstack([array, new_row])

                        # Filter rows where the first value of each row is 0
                        # filtered_rows = array[array[:, 0] == 0]
                        # # Sum the values in the third column of the filtered rows
                        # sum_of_third_column = np.sum(filtered_rows[:, 2])
                        # array_list.append(array)
 
                        list_probs.append(np.vstack(transitions_t0_t1.tocsr())) # Convert to Compressed Sparse Row format for efficient arithmetic operations

                        # list_probs.append(np.vstack(arrays))
                        
                        print(number)

                    np.save('probabilities_np.npy',list_probs)
                    return list_probs


    def get_state_transition_model_almost_there(self, num_actions: int, state_space: dict, time_step:int, num_years:int, material_probability_matrices:np.array, action_matrices:np.array, health_age_state_tansition_matrix:np.array, health_age_state_space:np.array):
        
        num_damage_states = 3  # good, medium, bad
        action_one_hot_enc = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])

        prob_matrix = np.zeros((num_actions, len(state_space), len(state_space)), dtype=np.float16)


        for action in range(num_actions):
            print('0')
            for current_state_index, current_state in enumerate(state_space):
                if current_state[0] == num_years:
                    prob_matrix[action][current_state_index][current_state_index] = 1
                    continue
                if action == 0:
                    # Find indices where the first value is equal to current_state[0] + time_step
                    # and values 4, 5, and 6 of state_space are greater than values 4, 5, and 6 of the current state
                    future_states_indices = np.where((state_space[:, 0] == current_state[0] + time_step) &
                                                    (state_space[:, 4] > current_state[4]) &
                                                    (state_space[:, 5] > current_state[5]) &
                                                    (state_space[:, 6] > current_state[6]))[0]

                elif action == 1: 
                    # Find indices where the first value is equal to current_state[0] + time_step and the fourth value is 0
                    future_states_indices = np.where((state_space[:, 0] == current_state[0] + time_step) & (state_space[:, 4] == 0))[0]

                elif action == 2: 
                    # Find indices where the first value is equal to current_state[0] + time_step and the fourth value is 0
                    future_states_indices = np.where((state_space[:, 0] == current_state[0] + time_step) & (state_space[:, 5] == 0))[0]

                else : 
                    # Find indices where the first value is equal to current_state[0] + time_step and the fourth value is 0
                    future_states_indices = np.where((state_space[:, 0] == current_state[0] + time_step) & (state_space[:, 6] == 0))[0]


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
                        prob_matrix[action][current_state_index][future_state_index] = new_probability

        # Normalize each row
        for action in range(num_actions):
            row_sums = prob_matrix[action].sum(axis=1)
            non_zero_sums = np.where((row_sums != 0) & (row_sums != 1))[0]

            for row_index in non_zero_sums:
                prob_matrix[action][row_index] /= row_sums[row_index]

        return prob_matrix


    
    def get_state_transition_model(self, num_actions: int, state_space: dict, time_step:int, num_years:int, material_probability_matrices:np.array, action_matrices:np.array, health_age_state_tansition_matrix:np.array, health_age_state_space:np.array):

        num_damage_states = 3  # good, medium, bad
        action_one_hot_enc = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])

        # Initialize lists to store sparse matrices for each action
        sparse_matrices = [lil_matrix((len(state_space), len(state_space)), dtype=np.float16) for _ in range(num_actions)]

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

                else : 
                    # Find indices where the first value is equal to current_state[0] + time_step and the fourth value is 0
                    future_states_indices = np.where((state_space[:, 0] == current_state[0] + time_step) &
                                                     (state_space[:, 4] == current_state[4]+time_step) &
                                                     (state_space[:, 5] == current_state[5]+time_step) &
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


        # Save the lil_matrix to a file
        # save_npz('sparse_matrix.npz', sparse_matrices)
        return sparse_matrices




    def get_state_transition_model_multiprocessors(self,num_actions: int, state_space: dict, time_step:int, num_years:int, material_probability_matrices:np.array, action_matrices:np.array, health_age_state_tansition_matrix:np.array, health_age_state_space:np.array):
        num_damage_states = 3  # good, medium, bad
        action_one_hot_enc = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])

        print('Transition model is starting')        
        with Pool() as pool:
            sparse_matrices = pool.starmap(process_action, [(action, state_space, time_step, num_years, num_damage_states, action_one_hot_enc, health_age_state_tansition_matrix, health_age_state_space) for action in range(num_actions)])

        # Save the sparse matrices
        for i, sparse_matrix in enumerate(sparse_matrices):
            # Convert Lil format sparse matrix to CSR format
            sparse_matrix_csr = sparse_matrix.tocsr()
            # Save the CSR format sparse matrix
            save_npz(f'sparse_matrix_action_{i}.npz', sparse_matrix_csr)
        
        print('Transition model is finished')

        return sparse_matrices





    def calculate_probability_array_for_state(self,current_state,action): 

        # Get info of state
        state_info = self.state_space[current_state]
        time_of_state = state_info[0]
        future_time_of_state = time_of_state + self.time_step

         #Make array to store data
        new_probability_array = np.zeros((1,len(self.state_space)))

        if time_of_state!= self.num_years:
            #Find the state index in the transition matrix
            #get the state 0
            states_t0 = self.state_space[self.state_space[:, 0] == 0]
            # Find the row where the values of columns 1 to 7 are the same as the state values (minus the time part)
            target_row_index = np.where((states_t0[:, 1:7] == state_info[1:]).all(axis=1))[0]
            index_of_current_state_in_probability_matrix = target_row_index[0]

            #Get the next_state slices from the state_space. These are the only states that are not going to be zero     
            # Find the indices where the first value of each row is the next timestep
            indexes = np.where(self.state_space[:, 0] == future_time_of_state)[0]
            first_index = indexes[0]
            last_index = indexes[-1] + 1
            index_places_to_populate_with_data = new_probability_array[first_index:last_index] 

            # Load the saved sparse matrices
            list_probs = np.load('probabilities_np.npy', allow_pickle=True)
            probability_array_for_action = list_probs[action] 
            probability_row_for_state_action = probability_array_for_action[index_of_current_state_in_probability_matrix].item()
            # Convert the sparse row to a dense array
            dense_row = probability_row_for_state_action.toarray()

            # Assign the values to columns corresponding to the next state space
            new_probability_array[:, first_index:last_index] = dense_row
        else: 
            #Make array to store data
            new_probability_array[0][current_state]= 1
            sm = np.sum(new_probability_array)

            # print('bla')
        
        return new_probability_array


    def get_state_transition_model0(self, num_actions: int, state_space: dict, time_step: int, num_years: int, material_probability_matrices: np.array, action_matrices: np.array):
        num_damage_states = 3  # good, medium, bad

        action_one_hot_enc = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
        list_probs = []

        for action in range(num_actions): 
            arrays = []
            states_t0 = state_space[state_space[:, 0] == 0]
            states_t1 = state_space[state_space[:, 0] == time_step]

            transitions_t0_t1 = lil_matrix((len(states_t0), len(states_t1)))

            for i, current_state in enumerate(states_t0):
                for j, future_state in enumerate(states_t1):
                    future_states_probabilities = np.zeros(num_damage_states)

                    for k in range(num_damage_states):
                        component_action = action_one_hot_enc[action][k]
                        current_component_st = int(current_state[k + 1])
                        future_component_st = int(future_state[k + 1])
                        current_component_age = int(current_state[k + 4])
                        future_component_age = int(future_state[k + 4])
                        age_to_index_convertion = int(current_component_age / time_step) if current_component_age != 0 else 0

                        if component_action == 0:  # Do nothing
                            if future_component_age == current_component_age + time_step:
                                probability = material_probability_matrices[age_to_index_convertion][current_component_st][future_component_st]
                            else:
                                probability = 0
                        else:  # Renovate
                            probability = action_matrices[current_component_st][future_component_st]
                        future_states_probabilities[k] = probability
                    total_prob = np.prod(future_states_probabilities)
                    transitions_t0_t1[i, j] = total_prob

            arrays.append(transitions_t0_t1.tocsr())
            list_probs.append(arrays)

        np.save('probabilities_np.npy', list_probs)
        return list_probs


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

    def get_reward_old(self, action: int, current_state: int) -> float:
        action_costs = self.renovation_costs[action]

        energy_demand_roof = self.energy_demand_nominal[0] * (1 + self.degradation_rates[self.state_space[current_state][1]])
        energy_demand_wall = self.energy_demand_nominal[1] * (1 + self.degradation_rates[self.state_space[current_state][2]])
        energy_demand_cellar = self.energy_demand_nominal[2] * (1 + self.degradation_rates[self.state_space[current_state][3]])
        total_energy_demand = energy_demand_roof + energy_demand_wall + energy_demand_cellar
        
        energy_bills = House.energy2euros(total_energy_demand)
        energy_bills = energy_bills * self.house_size_m2

        net_cost = action_costs + energy_bills
        reward = -net_cost
        if  self.state_space[current_state][0] ==  self.num_years: 
            reward = 0
        return reward

#Calculate Reward according to the imported data
    def get_reward_present(self, action: int, current_state: int) -> float:
        action_costs = self.renovation_costs[action]
        state_name = self.state_space[current_state][1:]
        total_energy_demand = self.kwh_per_state[state_name]
        # if total_energy_demand > 210:
        #     total_energy_demand = total_energy_demand * 2
        energy_bills = House.energy2euros(total_energy_demand)
        energy_bills = (energy_bills * self.house_size_m2)*self.time_step
        # if total_energy_demand > 210:
        #      energy_bills = 100000
        # if  self.state_space[current_state][0] ==  0: 
        #     net_cost = action_costs + (energy_bills)
        # else:
        #     net_cost = action_costs + (energy_bills*self.time_step)
        net_cost = action_costs + (energy_bills*self.time_step)
        reward = -net_cost

        # place zero reward to final states /absorbing states  
        if  self.state_space[current_state][0] ==  self.num_years: 
            reward = 0
        return reward

    def get_reward(self, action: int, current_state: int) -> float:
            
            action_costs = self.renovation_costs[action]
            state_name = self.state_space[current_state][1:4]
            total_energy_demand = self.kwh_per_state[tuple(state_name)]
            # if total_energy_demand > 210:
            #     total_energy_demand = total_energy_demand * 2
            energy_bills = House.energy2euros(total_energy_demand)
            energy_bills = (energy_bills * self.house_size_m2)*self.time_step
            # if total_energy_demand > 210:
            #      energy_bills = 100000
            # if  self.state_space[current_state][0] ==  0: 
            #     net_cost = action_costs + (energy_bills)
            # else:
            #     net_cost = action_costs + (energy_bills*self.time_step)
            net_cost = action_costs + (energy_bills*self.time_step)
            reward = -net_cost

            # place zero reward to final states /absorbing states  
            if  self.state_space[current_state][0] ==  self.num_years: 
                reward = 0
            return reward 

    def get_state_electricity_bill(self,state_space,kwh_per_state):
        energy_bills = {}
        for current_state in state_space:
            current_state = tuple(current_state)
            state_name = tuple(current_state[1:4])
            total_energy_demand = kwh_per_state[state_name]
            energy_bills[current_state] = total_energy_demand

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

    def get_transition_probs_old(self, current_state: int, action: int, time: int):
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
        

        for next_state in range(self.num_states):
            prob = self.state_transition_model[action][current_state][next_state]
            reward = self.get_reward(action=action, current_state=current_state)
            transition_probabilities.append((prob, next_state, reward))

        return transition_probabilities

    def get_transition_probs_newer(self, current_state: int, action: int, time: int):
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
        transition_probabilities = np.zeros((self.num_states, 3))
        probability_array = self.calculate_probability_array_for_state(current_state =current_state ,action = action)[0]
        for next_state in range(self.num_states):
            prob = probability_array[next_state]
            reward = self.get_reward(action=action, current_state=current_state)
            transition_probabilities[next_state, :] = [prob, next_state, reward]

        return transition_probabilities

    def get_transition_probs(self, current_state: int, action: int, time: int):
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
        transition_probabilities = np.zeros((self.num_states, 3))
        # Convert the sparse matrix to a dense matrix
        probability_array = self.state_transition_model[action][current_state].toarray()[0]
        for next_state in range(self.num_states):
            prob = probability_array[next_state]
            reward = self.get_reward(action=action, current_state=current_state)
            transition_probabilities[next_state, :] = [prob, next_state, reward]

        return transition_probabilities

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

    def step_old(self, action):
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
        p= self.calculate_probability_array_for_state(current_state =self.current_state ,action =action)[0]
        try:
            next_state = np.random.choice(self.num_states, p=p)

        except Exception as e:
            # Handle the specific exception if it occurs
            raise RuntimeError("The command failed: " + str(e) + ":" + str(p))


        
        # Calculate state reward
        
        reward = self.get_reward(action, self.current_state)

        # Check if episode is done (time limit reached)
        done = self.time >= self.num_years
        self.current_state = next_state

        return next_state, reward, done, {}

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
            raise RuntimeError("The command failed: " + str(e) + ":" + str(p))


        
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


if __name__=="__main__":
    env = House()
#     tryout = env.health_age_state_tansition_matrix
# #     state= env.num_years
    
    probs = env.state_transition_model

#     p = env.get_transition_probs(current_state = 0, action= 1, time =5)
# #     bla  = env.calculate_probability_array_for_state(current_state =1458 ,action =0)[0]
#     sm = sum(bla)
#     env.reset()
#     env.current_state = state
#     step = env.step(action = 0)
#     sm = np.sum(bla)
#     # prob1 = env.material_probability_matrices

#     bla = env.get_reward(action=0, current_state =0)
#     bills = env.energy_bills
#     states = env.state_space
# #     s = states[25]
    # print(bla)
# #     print(bla)
    # transitions = env.get_state_transition_model
#     # state_space = env.state_space
#     # with open("state_space.pickle", "wb") as f:
#     #     # Write the dictionary to the file using pickle.dump()
#     #     pickle.dump(state_space, f)
    print('bla')


    # with open("state_space.csv", "w", newline="") as csvfile:
    #     # Create a CSV writer object
    #     csv_writer = csv.writer(csvfile)

    #     # Write the header row (optional)
    #     csv_writer.writerow(["Key", "Value"])

    #     # Write each key-value pair as a row in the CSV file
    #     for key, value in state_space.items():
    #         csv_writer.writerow([key, value])
    
