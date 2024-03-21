import gym
import numpy as np
from house import House


def value_iteration(env: House, discount_factor=0.9, theta=1e-9, max_iter=1000):
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
                transitions = env.get_transition_probs(state, action, 0)[0]
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
            transitions = env.get_transition_probs(state, action, 0)[0]
            for prob, next_state, reward in transitions:
                action_values[action] += prob * (reward + discount_factor * V[next_state])
        best_action = np.argmax(action_values)
        policy[state] = best_action

    return policy, V
