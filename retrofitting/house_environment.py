from gym import Env, spaces
import numpy as np


class House(Env):
    def __init__(self, house_size_m2: float = 120):
        super(House, self).__init__()

        self.state_space = House.get_state_space(num_damage_states=3)
        self.num_states = len(self.state_space)

        num_actions = 4
        ##### ACTIONS #####
        # 0,  # DO_NOTHING
        # 1,  # FIX_ROOF
        # 2,  # FIX_WALL
        # 3   # FIX_FACADE
        self.action_space = spaces.Discrete(num_actions)

        self.observation_space = spaces.Discrete(self.num_states)
        self.current_state = 0
        self.time = 0
        self.num_years = 50
        self.time_step = 5
        self.p ,self.change_matrix = self.get_transition_matrices()
        self.ages = [0,0,0]
        # self.state_transition_model = House.get_state_transition_model(num_actions=num_actions,state_space=self.state_space)
        self.house_size_m2 = house_size_m2

        # [cost_doNothing, cost_roof, cost_wall, cost_cellar]
        self.renovation_costs = np.array([0, 2000, 5000, 3000])  # TODO: should change according to m2

        # [roof, wall, cellar]
        self.energy_demand_nominal = [57, 95, 38]
        self.degradation_rates = [0.0, 0.2, 0.4]

    def one_hot_enc(self,action):
        # Get transition probabilities, next state, and reward
        action_one_hot_enc = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
        act = action_one_hot_enc[action]  
        return act
    
    @staticmethod
    def get_state_space(num_damage_states: int):
        state_space = {}
        state_number = 0

        for r_damage_state in range(num_damage_states):
            for w_damage_state in range(num_damage_states):
                for c_damage_state in range(num_damage_states):
                    state_space[state_number] = (r_damage_state, w_damage_state, c_damage_state)
                    state_number += 1
        return state_space

    def get_transition_matrices(self):
    # Load npy file
        p = np.load('transition_matrices_n.npy')
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
        if action != 0:
            foo = 666
        action_costs = self.renovation_costs[action]

        energy_demand_roof = self.energy_demand_nominal[0] * (1 + self.degradation_rates[self.state_space[current_state][0]])
        energy_demand_wall = self.energy_demand_nominal[1] * (1 + self.degradation_rates[self.state_space[current_state][1]])
        energy_demand_cellar = self.energy_demand_nominal[2] * (1 + self.degradation_rates[self.state_space[current_state][2]])
        total_energy_demand = energy_demand_roof + energy_demand_wall + energy_demand_cellar

        energy_bills = House.energy2euros(total_energy_demand)
        energy_bills = energy_bills * self.house_size_m2

        net_cost = action_costs + energy_bills
        reward = -net_cost

        return reward
    
    def calculate_system_probability(self, current_state, component_ages, probability_matrices,state_space,action):
        overall_probability = []

        act = self.one_hot_enc(action)
        
        # Iterate over all possible next states
        for next_state in range(len(state_space)):

            transition_probability = 1.0
            
            # Iterate over each component
            for i, age in enumerate(component_ages):
                # Get the probability matrix for the current component based on its age
                n = state_space[current_state]

                if act[i] == 0 :

                    probability_matrix = probability_matrices[:,:,i][n[i]]
                    
                    # Get the probability of transitioning from the current state to the next state for this component
                    transition_probability_component = probability_matrix[state_space[next_state][i]]
                else:
                    transition_probability_component = self.change_matrix[n[i]][state_space[next_state][i]]

                # Multiply the probabilities for all components
                transition_probability *= transition_probability_component

                # print(transition_probability)
            # Add the probability of this next state to the overall probability
            overall_probability.append(transition_probability)

            sum = 0
            for i in overall_probability: 
                sum +=i
            # print(overall_probability)
            # print(f'sum is {sum}')
            # if sum > 1 :
            #     print('we did wrong again?')
            # elif sum < 1: 
            #     print('sum<1')

        return overall_probability
    

    # def calculate_transition_probability(self,state :int ,next_state :int ,ages :list ,p :np.array, action:int,state_space):
    #     action_one_hot_enc = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    #     list = []
    #     # Initialize a variable to hold the sum
    #     total_product = 1
    #     for index,value in enumerate(state_space[state]): 
    #         next_value = state_space[next_state][index] 
    #         if action_one_hot_enc[action][index] == 0:
    #             component = p[:,:,ages[index]][value][next_value]
    #         else: 
    #             component = transition_matrix[value][next_value]
    #         list.append(component)
                
        # for num in list: 
        #      total_product *= num
        # # print(total_product)

        # return total_product
    

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
        
        act = self.one_hot_enc(action)

        pr = self.calculate_system_probability(current_state = current_state, component_ages =self.ages, probability_matrices = self.p,state_space=self.state_space, action = action)

        for next_state in range(self.num_states):
            # prob = self.state_transition_model[action][current_state][next_state]
            prob = pr[next_state]
            # prob = self.calculate_transition_probability(state =current_state,next_state =next_state,ages=self.ages,p = p,action = action,state_space=self.state_space)
            reward = self.get_reward(action=action, current_state=current_state)

            transition_probabilities.append((prob, next_state, reward))
        
        total_sum = 0

        # Iterate through the list and add each probability to the sum
        for prob, _, _ in transition_probabilities:
            total_sum += prob

        # Round the total_sum to three decimal places
        total_sum = round(total_sum, 3)

        if total_sum > 1:
            raise Exception('Probability is bigger than 1. We guessed wrong')

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