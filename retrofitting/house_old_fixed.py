from gym import Env, spaces
import numpy as np
import pandas as pd
import pickle 
import csv
from scipy.sparse import lil_matrix
from gamma_deterioration import matrices_gen 
import time
from scipy.sparse import csr_matrix

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
        self.num_years = 10
        self.time_step = 5
        self.action_space = spaces.Discrete(4)
        self.state_space = House.get_state_space(num_damage_states=self.num_damage_states,
                                                 num_years= self.num_years,
                                                 time_step = self.time_step)
        self.num_states = len(self.state_space)
        self.observation_space = spaces.Discrete(self.num_states)

        _, self.kwh_per_state = self.import_simulation_data(file_path='building_scenarios_copy.csv',no_windows=True)
        self.house_size_m2 = house_size_m2

        # [cost_doNothing, cost_roof, cost_wall, cost_cellar]
        self.renovation_costs = np.array([0, 13791, 16273, 5000])  # 

        # [roof, wall, cellar]
        # self.energy_demand_nominal = [57, 95, 38]
        self.degradation_rates = [0.0, 0.2, 0.5]
        # self.energy_bills = self.get_state_electricity_bill(state_space = self.state_space,kwh_per_state=self.kwh_per_state)
        self.material_probability_matrices,self.action_matrices = \
                                                self.import_gamma_probabilities(calculate_gamma_distribution_probabilities= True,
                                                step_size=self.time_step,SIMPLE_STUFF = True, 
                                                N = 1000, do_plot= False, T = self.num_years+self.time_step,
                                                save_probabilities = True)
        # self.state_transition_model = House.get_state_transition_model(num_actions=self.num_actions, state_space=self.state_space, time_step= self.time_step,num_years = self.num_years)
        # self.state_transition_model = self.get_state_transition_model(num_actions=self.num_actions,
        #                                                               state_space=self.state_space, 
        #                                                               time_step= self.time_step,
        #                                                               num_years = self.num_years,
        #                                                               material_probability_matrices= self.material_probability_matrices,
        #                                                               action_matrices=self.action_matrices)
 
####################################################################################################

    def get_state_space_old(num_damage_states: int, num_years: int, time_step: int):
        state_space = {}
        state_number = 0
        for time in range(0,num_years+time_step,time_step):
            for r_damage_state in range(num_damage_states):
                for w_damage_state in range(num_damage_states):
                    for c_damage_state in range(num_damage_states):
                        state_space[state_number] = (time,r_damage_state, w_damage_state, c_damage_state)
                        state_number += 1
        return state_space

#State space as a dictionary with ages 
    def get_state_space_with_ages(num_damage_states: int, num_years: int, time_step: int):
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
    
#State space as a numpy with ages and time
    def get_state_space(num_damage_states: int, num_years: int, time_step: int):
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
    def get_state_space_cleared(num_damage_states: int, num_years: int, time_step: int):
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

            ## Create a list to store keys to delete
            new_state_space = np.empty((0,7))
            for index,state in enumerate(state_space):
                if state[0] >= state[4] and state[0] >= state[5] and state[0] >= state[6] :
                   new_state_space = np.vstack([new_state_space, state])

            return state_space

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
    
    @staticmethod
    def get_state_transition_model_old(num_actions: int, state_space: dict, time_step:int, num_years:int):
        num_damage_states = 3  # good, medium, bad

        # Define transition model of components
        TRANSITION_MODEL = np.zeros((num_actions, num_damage_states, num_damage_states))
        TRANSITION_MODEL[0] = np.array([[0.8, 0.2, 0.0],
                                        [0.0, 0.8, 0.2],
                                        [0.0, 0.0, 1.0]])
        TRANSITION_MODEL[1] = np.array([[1, 0, 0],
                                        [1, 0, 0],
                                        [1, 0, 0]])

        # Calculate transition model of system
        STATE_TRANSITION_MODEL = np.zeros((num_actions, len(state_space), len(state_space)))
        action_one_hot_enc = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])

        for action in range(num_actions): 
            for key in state_space.keys(): # get state number
                row = []
                for future_key in state_space.keys(): # get future state number
                    future_states_probabilities = []
                    for i in range(num_damage_states): #get the damage state 
                        action_array = action_one_hot_enc[action][i] #get the action for that 
                        probability = TRANSITION_MODEL[action_array][state_space[key][i]][state_space[future_key][i]]
                        future_states_probabilities.append(probability)
                    new_probability = np.prod(future_states_probabilities)
                    row.append(new_probability)
                STATE_TRANSITION_MODEL[action][key] = row
        return STATE_TRANSITION_MODEL
    
    def get_state_transition_model_present(num_actions: int, state_space: dict, time_step:int, num_years:int):
            num_damage_states = 3  # good, medium, bad

            # Define transition model of components
            TRANSITION_MODEL = np.zeros((num_actions, num_damage_states, num_damage_states))
            TRANSITION_MODEL[0] = np.array([[0.8, 0.2, 0.0],
                                            [0.0, 0.8, 0.2],
                                            [0.0, 0.0, 1.0]])
            TRANSITION_MODEL[1] = np.array([[1, 0, 0],
                                            [1, 0, 0],
                                            [1, 0, 0]])

            # Calculate transition model of system
            STATE_TRANSITION_MODEL = np.zeros((num_actions, len(state_space), len(state_space)))
            action_one_hot_enc = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
            list2 = []
            for action in range(num_actions): 
                list = []
                for key in state_space.keys(): # get state number
                    row = []
                    for future_key in state_space.keys(): # get future state number
                        future_states_probabilities = []
                        for i in range(num_damage_states): #get the damage state 
                            action_array = action_one_hot_enc[action][i] #get the action for that 
                            probability = TRANSITION_MODEL[action_array][state_space[key][i+1]][state_space[future_key][i+1]]
                            future_states_probabilities.append(probability)
                        new_probability = np.prod(future_states_probabilities)
                        if state_space[future_key][0] != state_space[key][0]+time_step:
                            new_probability = 0 
                        if state_space[key][0] ==num_years and state_space[future_key]== state_space[key]:
                            new_probability = 1 
                            
                        row.append(new_probability)
                    STATE_TRANSITION_MODEL[action][key] = row
                    sum = np.sum(STATE_TRANSITION_MODEL[action][key])
                    list.append(sum)
                list2.append(list)

            return STATE_TRANSITION_MODEL
#newer, using np. still slow
    def get_state_transition_model_npversion(self, num_actions: int, state_space: dict, time_step:int, num_years:int,material_probability_matrices:np.array, action_matrices:np.array):
                num_damage_states = 3  # good, medium, bad

                # Calculate transition model of system
                action_one_hot_enc = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
                list_probs = []
                # number = 0

                for action in range(num_actions): 
                    # number+=1
                    arrays = []
                    number = 0
                    array = np.empty((0, 3))
                    for current_state_index,current_state in enumerate(state_space): # get state number
                        year = current_state[0]
                        # Find indices where the first value of each row equals to the next year from the state space that we are in now
                        indices = np.where(state_space[:, 0] == year+time_step)[0]
                        tic = time.time()
                        for future_state in indices: # get future state index number
                            future_key = state_space[future_state]  # get the tuple describing the next state 
                            future_states_probabilities = np.zeros(3)
                            if int(current_state[0]) != num_years:
                                for i in range(num_damage_states): #get the damage state 
                                    action_array = action_one_hot_enc[action][i] #get the action for that
                                    current_component_st = int(current_state[i+1]) # since the first number is the time, start with the second
                                    future_component_st = int(future_key[i+1])
                                    current_component_age = int(current_state[i+4]) 
                                    future_component_age = int(future_key[i+4])
                                    age_to_index_convertion = int(current_component_age/time_step) if current_component_age != 0 else 0
                                    if action_array == 0 :
                                        if future_component_age == current_component_age+time_step:
                                            probability = material_probability_matrices[age_to_index_convertion][current_component_st][future_component_st]
                                        else :
                                            probability = 0
                                    else: 
                                        probability = action_matrices[current_component_st][future_component_st]
                                    future_states_probabilities[i] = probability
                                
                                new_probability = np.prod(future_states_probabilities)   
                                if new_probability != 0:
                                    new_row = np.array([current_state_index,future_state,new_probability])
                                    arrays.append(new_row)
                                    # array = np.vstack([array, new_row])
                                toc = time.time()
                                print(toc - tic)
                            elif current_state[0] == num_years  and current_state == future_key:
                                new_probability = 1 # make sure that the final state can only go to itself and no other state 
                                new_row = np.array([current_state, future_state, new_probability])
                                arrays.append(new_row)
                        # array = np.vstack([array, new_row])

                    # Filter rows where the first value of each row is 0
                    # filtered_rows = array[array[:, 0] == 0]
                    # # Sum the values in the third column of the filtered rows
                    # sum_of_third_column = np.sum(filtered_rows[:, 2])
                    # array_list.append(array)
                    list_probs.append(np.vstack(arrays))
                    
                    print(number)

                np.save('probabilities_np.npy',list_probs)
                return list_probs

#newer, using np., a bit faster,wth sparse matrices 
    def get_state_transition_model(self, num_actions: int, state_space: dict, time_step:int, num_years:int,material_probability_matrices:np.array, action_matrices:np.array):
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

            sm = np.sum(new_probability_array)
            print('bla')
        
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

    def get_reward(self, action: int, current_state: int) -> float:
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
    
    def get_state_electricity_bill(self,state_space,kwh_per_state):
        energy_bills = {}
        for current_state in state_space.keys():
            state_name = state_space[current_state][1:4]
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
        transition_probabilities = []

        for next_state in range(self.num_states):
            prob = self.state_transition_model[action][current_state][next_state]
            reward = self.get_reward(action=action, current_state=current_state)
            transition_probabilities.append((prob, next_state, reward))

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
        next_state = np.random.choice(self.num_states, p=self.state_transition_model[action][self.current_state])

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
    # tyout = env.state_transition_model
    bla  = env.calculate_probability_array_for_state(current_state =1000 ,action =0)
    sm = np.sum(bla)
    # prob1 = env.material_probability_matrices

#     bla = env.get_reward(action=0, current_state =0)
#     bills = env.energy_bills
#     states = env.state_space
# #     s = states[25]
    # print(bla)
# #     print(bla)
#     transitions = env.get_state_transition_model
#     # state_space = env.state_space
#     # with open("state_space.pickle", "wb") as f:
#     #     # Write the dictionary to the file using pickle.dump()
#     #     pickle.dump(state_space, f)
#     print('bla')


    # with open("state_space.csv", "w", newline="") as csvfile:
    #     # Create a CSV writer object
    #     csv_writer = csv.writer(csvfile)

    #     # Write the header row (optional)
    #     csv_writer.writerow(["Key", "Value"])

    #     # Write each key-value pair as a row in the CSV file
    #     for key, value in state_space.items():
    #         csv_writer.writerow([key, value])
    
