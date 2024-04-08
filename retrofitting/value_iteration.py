import gym
import numpy as np
from retrofitting.house_environment import House


def value_iteration(env: House, discount_factor=0.97):
    # Initialise values of all states with 0
    value_function = np.zeros(env.num_states)
    q_values = np.zeros((env.num_states, 4))
    optimal_action = np.zeros(env.observation_space.n, dtype=int)

    delta_threshold = 1e-20
    delta = 10

    num_iterations = 5000
    # Fixed point iteration
    while delta > delta_threshold:
        num_iterations += 1
        delta = 0

        
        for idx_state in range(env.num_states):
            old_value = value_function[idx_state]

            # store Q values of all actions
            for action in range(4):
                q_val = 0
                transition_probs, _ = env.get_transition_probs(current_state=idx_state, action=action, time=0)
                for current_tuple in transition_probs:
                    prob, next_state, reward = current_tuple
                    q_val += prob * (reward + discount_factor * value_function[next_state])
                q_values[idx_state, action] = q_val

            # value_function[state] = max(q_values)
            idx = np.random.choice(np.flatnonzero(q_values[idx_state, :] == max(q_values[idx_state, :])))
            optimal_action[idx_state] = idx
            value_function[idx_state] = q_values[idx_state, idx]

            delta = max([delta, np.abs(old_value - value_function[idx_state])])

    return optimal_action, value_function, num_iterations




# def extract_policy(env,value_table, gamma = 1.0):
#     policy = np.zeros(env.observation_space.n)
#     for state in range(env.observation_space.n):
#         Q_table = np.zeros(env.action_space.n)
#         for action in range(env.action_space.n):
#             for next_sr in env.P[state][action]:
#                 trans_prob, next_state, reward_prob, _ = next_sr
#                 Q_table[action] += (trans_prob * (reward_prob + gamma *
#                 value_table[next_state]))
#                 policy[state] = np.argmax(Q_table)
#                 return policy
#     optimal_value_function = value_iteration(env=env,gamma=1.0)
#     optimal_policy = extract_policy(optimal_value_function, gamma=1.0)
#     print(optimal_policy)