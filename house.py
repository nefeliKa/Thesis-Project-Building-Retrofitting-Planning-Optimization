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

        self.state_transition_model = House.get_state_transition_model(num_actions=num_actions,
                                                                       state_space=self.state_space)
        self.house_size_m2 = house_size_m2

        # [cost_doNothing, cost_roof, cost_wall, cost_cellar]
        self.renovation_costs = np.array([0, 2000, 5000, 3000])  # TODO: should change according to m2

        # [roof, wall, cellar]
        self.energy_demand_nominal = [57, 95, 38]
        self.degradation_rates = [0.0, 0.2, 0.4]
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

    @staticmethod
    def get_state_transition_model(num_actions: int, state_space: dict, ):
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
            for key in state_space.keys():
                row = []
                for future_key in state_space.keys():
                    future_states_probabilities = []
                    for i in range(num_damage_states):
                        action_array = action_one_hot_enc[action][i]
                        probability = TRANSITION_MODEL[action_array][state_space[key][i]][state_space[future_key][i]]
                        future_states_probabilities.append(probability)
                    new_probability = np.prod(future_states_probabilities)
                    row.append(new_probability)
                STATE_TRANSITION_MODEL[action][key] = row
        return STATE_TRANSITION_MODEL

    @staticmethod
    def energy2euros(num_of_kwh: float) -> float:
        price_kwh = 0.35
        return price_kwh * num_of_kwh

    def get_reward(self, action: int, current_state: int) -> float:
        action_costs = self.renovation_costs[action]

        energy_demand_roof = self.energy_demand_nominal[0] * (1 + self.degradation_rates[self.state_space[current_state][0]])
        energy_demand_wall = self.energy_demand_nominal[1] * (1 + self.degradation_rates[self.state_space[current_state][1]])
        energy_demand_cellar = self.energy_demand_nominal[2] * (1 + self.degradation_rates[self.state_space[current_state][2]])
        total_energy_demand = energy_demand_roof + energy_demand_wall + energy_demand_wall

        energy_bills = (House.energy2euros(energy_demand_roof) +
                        House.energy2euros(energy_demand_wall) +
                        House.energy2euros(energy_demand_cellar))
        energy_bills = energy_bills * self.house_size_m2

        net_cost = action_costs + energy_bills
        reward = -net_cost

        return reward

    def get_transition_probs(self, current_state: int, action: int, time: int) -> tuple[list, int]:
        """
        MDP model for the environment.
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

        # Check if episode should terminate due to time limit
        if time >= self.num_years:
            return transition_probabilities, time

        for s_ in range(self.num_states):
            next_state = s_
            prob = self.state_transition_model[action][current_state][s_]
            reward = self.get_reward(action=action, current_state=current_state)

            transition_probabilities.append((prob, next_state, reward))

        time += self.time_step
        return transition_probabilities, time

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

        return next_state, reward, done, {}

    def render(self, mode='human'):
        # Optionally render the environment
        pass

    def close(self):
        # Clean up resources, if any
        pass

