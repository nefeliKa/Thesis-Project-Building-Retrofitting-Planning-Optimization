import gym
import numpy as np
from house_old_fixed import House
import matplotlib.pyplot as plt
import seaborn as sns


def value_iteration(env: House):
    discount_factor = 0.97**env.time_step
    # Initialise values of all states with 0
    value_function = np.zeros(env.num_states)
    q_values = np.zeros((env.num_states, 4))
    optimal_action = np.zeros(env.observation_space.n, dtype=int)

    delta_threshold = 1e-20
    delta = 10

    num_iterations = 0
    # Fixed point iteration
    while delta > delta_threshold:
        num_iterations += 1
        delta = 0

        # ignore terminal states, since value is 0
        for idx_state in range(env.num_states):
            # if idx_state: 
            #     continue
            old_value = value_function[idx_state]

            # store Q values of all actions
            for action in range(4):
                q_val = 0
                transition_probs = env.get_transition_probs(current_state=idx_state, action=action, time=0)
                for current_tuple in transition_probs:
                    prob, next_state, reward = current_tuple
                    q_val += prob * (reward + discount_factor * value_function[next_state])
                q_values[idx_state, action] = q_val

            # value_function[state] = max(q_values)
            idx = np.random.choice(np.flatnonzero(q_values[idx_state, :] == max(q_values[idx_state, :])))
            optimal_action[idx_state] = idx
            value_function[idx_state] = q_values[idx_state, idx]

            delta = max([delta, np.abs(old_value - value_function[idx_state])])
    
    print('Value_iteration_finished')
    return optimal_action, value_function, num_iterations



# if __name__ == "__main__":
#     env = House()
#     # transitions = env.get_state_transition_model
#     # print('bla')
#     TIME_HORIZON = int(env.num_years/env.time_step)+ 1
#     length = len(env.state_space)
#     num_damage_states = 27
#     # state_space = np.array(env.state_space.values())
#     # state_space.reshape(TIME_HORIZON, num_damage_states)
#     optimal_action, value_function, num_iterations = value_iteration(env)
#     reshaped_value_func = value_function.reshape(TIME_HORIZON, num_damage_states)
#     optimal_action.reshape(TIME_HORIZON, num_damage_states)

#     print(f"Value of states: [time_horizon, num_states] \n {np.around(value_function.reshape(TIME_HORIZON, num_damage_states),3)}")

# #     _value_func_2d = value_function.reshape(TIME_HORIZON, num_damage_states)
# #     _optimal_actions_2d = optimal_action.reshape(TIME_HORIZON, num_damage_states)

# #     fig, ax = plt.subplots(figsize=(5, 4))
#     sns.heatmap(_value_func_2d, cmap="inferno", annot=_value_func_2d, annot_kws={'fontsize': 8}, cbar_kws={"shrink": .8})

#     ax.set_xlabel('Damage state')
#     ax.set_ylabel('Time')
#     ax.set_title('Values of states')

#     plt.show()
    
#     print('bla')