from gym import Env
from house_environment import House
from value_iteration import value_iteration
import matplotlib.pyplot as plt
import numpy as np

#TODO 
#check how many episodes I have to run it to have a good estimate
#check the results for different policies
#run 10000 episodes to evaluate policy because there is a stochasticisity 
#next time: priority difference of deterioration rate over time 
#next time : do nothing - evaluate the rewards according to that 


def run_episodes(env: House, policy: list = None, num_episodes: int = 100):
    total_costs_episodes = []

    num_time_steps = env.num_years // env.time_step
    rewards_all_episodes = np.zeros((num_episodes, num_time_steps))
    states_all_episodes = np.zeros((num_episodes, num_time_steps))
    actions_all_episodes = np.zeros((num_episodes, num_time_steps))

    # Run episodes
    for episode in range(num_episodes):
        print("Episode:", episode + 1)
        state = env.reset()
        done = False
        time_idx = 0
        total_cost = 0
        rewards_current_episode = np.zeros(num_time_steps)
        states_current_episode = np.zeros(num_time_steps)
        actions_current_episode = np.zeros(num_time_steps)
        while not done:
            if policy is None:
                action = env.action_space.sample()  # Random action for demonstration
            else:
                action = policy[state]
            next_state, reward, done, _ = env.step(action)
            print(f"\tTime step: {time_idx} --> Current State: {state}: {env.state_space[state]} | Action: {action} | Next State: {next_state} | Reward: {reward:.2f}")
            total_cost += abs(reward)
            rewards_current_episode[time_idx] = (abs(np.round(reward)))
            states_current_episode[time_idx] = state
            actions_current_episode[time_idx] = action
            state = next_state
            time_idx += 1

        rewards_all_episodes[episode, :] = rewards_current_episode
        states_all_episodes[episode, :] = states_current_episode
        actions_all_episodes[episode, :] = actions_current_episode
        print("Episode finished.\n")

        total_costs_episodes.append(total_cost)

    return total_costs_episodes, rewards_all_episodes, states_all_episodes,actions_all_episodes


def plot_histogram_comparisons(data1, data2):
    # Create a new figure with 1 row and 2 columns of subplots
    plt.figure(figsize=(10, 5))

    # Plot histogram for data1
    plt.hist(data1, bins='auto', color='blue', alpha=0.7, label='Do-nothing policy')

    # Plot histogram for data2
    plt.hist(data2, bins='auto', color='red', alpha=0.7, label='Optimal policy')

    plt.xlabel("Cost [€]")
    plt.ylabel("Frequency")
    plt.title("Histogram Comparison")
    plt.legend()

    plt.show()


def plot_costs_for_policy(env: House, policy: list, rewards: np.ndarray, states: np.ndarray, plot_title: str):
    actions = {0: 'DN', 1: 'R', 2: 'W', 3: 'C'}
    colors = {'DN': 'black', 'R': 'red', 'W': 'green', 'C': 'blue'}
    # labels = {'DN': 'do nothing', 'R': 'roof', 'W': 'wall', 'C': 'cellar'}

    time_axis = np.arange(0, env.num_years, env.time_step)
    plt.plot(time_axis, rewards, marker='o', linestyle='-')
    for i, (xi, yi) in enumerate(zip(time_axis, rewards)):
        k = int(states[i])
        string = actions[policy[k]]  #GOTCHA . Here should be the action that was taken in that state, not the generla policy
        plt.text(xi, yi, string, color=colors[string], weight='bold', ha='center', va='bottom')

    for i, (xi, yi) in enumerate(zip(time_axis, rewards)):
        string = str(int(states[i]))
        plt.text(xi, yi, string, weight='bold', ha='right', va='top')

    plt.ylabel("Costs [€]")
    plt.xlabel("Years")
    plt.title(f"{plot_title} policy")

    plt.grid()
    plt.show()
#TODO :
    #Plot a histogram with the energy bills, the actions that are taken and the time 
    #Clean the means/variance code and make it into a function 
    #Make comments under each function 

        


if __name__ == "__main__":

    env = House()
    zero_policy = [0 for _ in range(env.observation_space.n)] #create a zero policy
    # optimal_policy, optimal_value,_ = value_iteration(env)

    # Evaluate "do-nothing" policy
    # total_costs_zero_policy, rewards_all_episodes_zero_policy, states_all_episodes_zero_policy = run_episodes(env=env, policy=zero_policy, num_episodes=100)  # run the zero policy

    # plot costs/policy/states for 1st episode
    # plot_costs_for_policy(env, zero_policy, rewards_all_episodes_zero_policy[0], states_all_episodes_zero_policy[0], plot_title='"Do nothing"')
    # total_costs_random_policy = run_episodes(env=env, policy=None) #run a random policy

    # Evaluate using value iteration
    env.reset()
    optimal_policy, optimal_value, num_iterations = value_iteration(env)
    total_costs_value_iteration, rewards_all_episodes_value_iteration, states_all_episodes_value_iteration,actions_all_episodes = run_episodes(env=env, policy=optimal_policy, num_episodes=100)

    # plot costs/policy/states for 1st episode
    for n in range(0,10):
        plot_costs_for_policy(env, optimal_policy, rewards_all_episodes_value_iteration[n], states_all_episodes_value_iteration[n], plot_title='Value Iteration')

    print(f"Number of iterations for optimal policy: {num_iterations}")
    print(f"The optimal policy is: {optimal_policy}")
    # plot_histogram_comparisons(total_costs_zero_policy, total_costs_value_iteration)

    #find the mean and the variance of each policy'
    #mean = policy1+policy2+policy3+...policyn / n
    #subtract the mean from each policy variance :v1= (mean - policy1)^2
    #variance = v1 + v2 +v3 +...vn / n
    # mean_zero_policy = sum(total_costs_zero_policy)/ len(total_costs_zero_policy)
    # mean_zero_variance = []
    # for value in range(len(total_costs_zero_policy)): 
    #     variance = (mean_zero_policy - total_costs_zero_policy[value])**2
    #     mean_zero_variance.append(variance)
    # mean_zero_variance = sum(mean_zero_variance)/len(mean_zero_variance)
    # mean_zero_variance = np.sqrt(mean_zero_variance)

    #find mean and variance for optimal policy
    mean_optimal_policy = sum(total_costs_value_iteration)/ len(total_costs_value_iteration)
    mean_optimal_variance = []
    for value in range(len(total_costs_value_iteration)): 
        optimal_variance = (mean_optimal_policy - total_costs_value_iteration[value])**2
        mean_optimal_variance.append(optimal_variance)
    mean_optimal_variance = sum(mean_optimal_variance)/len(mean_optimal_variance)
    mean_optimal_variance = np.sqrt(mean_optimal_variance)

    print(f"The mean return of optimal policy is: {mean_optimal_policy}. The variance is:{mean_optimal_variance} ")

    # print(f"The mean return of zero policy is: {mean_zero_policy}. The variance is:{mean_zero_variance} ")



