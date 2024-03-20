import numpy as np
import gym
from gym import spaces
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib

# Define environment parameters
energy_demand_nominal = [57, 95, 38]
degradation_rates = [0.0, 0.2, 0.4]
house_size_m2 = 120
renovation_costs = np.array([0, 2000, 5000, 3000])
ROOF_REPLACE_COST = -50
CELLAR_REPLACE_COST = -50
WALL_REPLACE_COST = -50
REWARDS = [0, ROOF_REPLACE_COST, WALL_REPLACE_COST, CELLAR_REPLACE_COST]
PENALTY = -500

# Define state space
num_damage_states = 3
state_space = {}
key = 0

for r_damage_state in range(num_damage_states):
    for w_damage_state in range(num_damage_states):
        for c_damage_state in range(num_damage_states):
            state_space[key] = (r_damage_state, w_damage_state, c_damage_state)
            key += 1

# Define action space
DO_NOTHING = 0
FIX_ROOF = 1
FIX_WALL = 2
FIX_FACADE = 3
action_space = [DO_NOTHING, FIX_ROOF, FIX_WALL, FIX_FACADE]
num_actions = len(action_space)

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


# Define discount factor
DISCOUNT_FACTOR = 0.9


def energy2euros(num_of_kwh: float) -> float:
    price_kwh = 0.35
    return price_kwh * num_of_kwh


class RenovationEnv(gym.Env):
    def __init__(self):
        super(RenovationEnv, self).__init__()

        # Define action and observation space
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Discrete(len(state_space))
        self.current_state = 0

        # Initialize time
        self.time = 0

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
        #transition_probs, next_state, reward = self.MDP_model(self.state, action, self.time)

        # Update time
        self.time += 5

        # Choose next state based on transition probabilities
        next_state = np.random.choice(len(state_space), p=STATE_TRANSITION_MODEL[action][self.current_state])

        # Calculate state reward
        def get_reward(action, current_state):
            action_costs = renovation_costs[action]

            energy_demand_roof = energy_demand_nominal[0] * (1 + degradation_rates[state_space[current_state][0]])
            energy_demand_wall = energy_demand_nominal[1] * (1 + degradation_rates[state_space[current_state][1]])
            energy_demand_cellar = energy_demand_nominal[2] * (1 + degradation_rates[state_space[current_state][2]])
            total_energy_demand = energy_demand_roof + energy_demand_wall + energy_demand_wall

            energy_bills = (energy2euros(energy_demand_roof) +
                            energy2euros(energy_demand_wall) +
                            energy2euros(energy_demand_cellar))
            energy_bills = energy_bills * house_size_m2

            net_cost = action_costs + energy_bills
            reward = -net_cost

            return reward

        reward = get_reward(action,self.current_state)



        # Check if episode is done (time limit reached)
        done = self.time >= 50

        return next_state, reward, done, {}

    def render(self, mode='human'):
        # Optionally render the environment
        pass

    def close(self):
        # Clean up resources, if any
        pass

    def MDP_model(self, current_state, action, time):
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
        output = []

        # Check if episode should terminate due to time limit
        if time >= 50:
            return output, time

        # Calculate state reward
        def get_reward(action, current_state):
            action_costs = renovation_costs[action]

            energy_demand_roof = energy_demand_nominal[0] * (1 + degradation_rates[state_space[current_state][0]])
            energy_demand_wall = energy_demand_nominal[1] * (1 + degradation_rates[state_space[current_state][1]])
            energy_demand_cellar = energy_demand_nominal[2] * (1 + degradation_rates[state_space[current_state][2]])
            total_energy_demand = energy_demand_roof + energy_demand_wall + energy_demand_wall

            energy_bills = (energy2euros(energy_demand_roof) +
                            energy2euros(energy_demand_wall) +
                            energy2euros(energy_demand_cellar))
            energy_bills = energy_bills * house_size_m2

            net_cost = action_costs + energy_bills
            reward = -net_cost

            return reward

        for s_ in range(len(state_space)):
            next_state = s_
            prob = STATE_TRANSITION_MODEL[action][current_state][s_]
            reward = get_reward(action=action, current_state=current_state)

            output.append((prob, next_state, reward))

        time += 5
        return output, time





# Value Iteration
def value_iteration(env, discount_factor=0.9, theta=1e-9, max_iter=1000):
    """
    Value Iteration Algorithm.
    Parameters
    ----------
    env : gym.Env
        The Gym environment.
    discount_factor : float
        The discount factor for future rewards.
    theta : float
        A very small positive number to decide whether to stop iterations.
    max_iter : int
        Maximum number of iterations.
    Returns
    -------
    policy : np.array
        Array of optimal action for each state.
    V : np.array
        Array of value for each state.
    """
    # Initialize value function
    V = np.zeros(env.observation_space.n)

    for i in range(max_iter):
        delta = 0
        for state in range(env.observation_space.n):
            # Calculate the expected value of each action
            action_values = np.zeros(env.action_space.n)
            for action in range(env.action_space.n):
                transitions = env.MDP_model(state, action, 0)[0]
                for prob, next_state, reward in transitions:
                    action_values[action] += prob * (reward + discount_factor * V[next_state])
            # Choose the best action
            best_action_value = np.max(action_values)
            # Calculate the difference between current value and new value
            delta = max(delta, np.abs(best_action_value - V[state]))
            # Update the value function
            V[state] = best_action_value
        # Check if the change in value function is small enough
        if delta < theta:
            break

    # Extract policy
    policy = np.zeros(env.observation_space.n, dtype=int)
    for state in range(env.observation_space.n):
        action_values = np.zeros(env.action_space.n)
        for action in range(env.action_space.n):
            transitions = env.MDP_model(state, action, 0)[0]
            for prob, next_state, reward in transitions:
                action_values[action] += prob * (reward + discount_factor * V[next_state])
        best_action = np.argmax(action_values)
        policy[state] = best_action

    return policy, V


# Create an instance of the environment
env = RenovationEnv()

# Solve the environment using value iteration
optimal_policy, optimal_value = value_iteration(env)
print(optimal_policy)

# Extracting value function and optimal action
value_function = optimal_value
optimal_action = optimal_policy

# Assuming TIME_HORIZON and num_damage_states are defined in your code
TIME_HORIZON = 10  # Example value
num_damage_states = 27  # Example value




import matplotlib.pyplot as plt

# Assuming optimal_action is a 1D array containing the optimal action for each state
# Define the states and state tuples
states = list(range(len(optimal_action)))
state_tuples = [state_space[state] for state in states]


# Plotting the optimal policy for one episode using a line plot with state tuples as labels
plt.figure(figsize=(12, 6))
plt.plot(states, optimal_action, marker='o', linestyle='-')
plt.xlabel('State')
plt.ylabel('Optimal Action')
plt.title('Optimal Policy for One Episode')
plt.xticks(states, [f'State {i} = {state_tuples[i]}' for i in states], rotation=45, ha='right')
plt.grid(True)
plt.tight_layout()
plt.show()


# Create an instance of the environment
env = RenovationEnv()

avg_total_cost = 0

# Run episodes
for episode in range(5):
    print("Episode:", episode + 1)
    state = env.reset()
    done = False
    time_idx = 0
    total_cost = 0
    while not done:
        # action = env.action_space.sample()  # Random action for demonstration
        action = optimal_policy[time_idx]
        next_state, reward, done, _ = env.step(action)
        print("Current State:", state, "Action:", action, "Next State:", next_state, "Reward:", reward)
        total_cost += abs(reward)
        state = next_state
        time_idx +=1
    print("Episode finished.")

    avg_total_cost += total_cost

print(f"Avg total cost: {avg_total_cost/5}")

