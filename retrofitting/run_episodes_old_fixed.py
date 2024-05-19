from gym import Env
from house_old_fixed import House
from value_iteration_old_fixed import value_iteration
import matplotlib.pyplot as plt
import numpy as np

#TODO 
#check how many episodes I have to run it to have a good estimate
#check the results for different policies
#run 10000 episodes to evaluate policy because there is a stochasticisity 
#next time: priority difference of deterioration rate over time 
#next time : do nothing - evaluate the rewards according to that 


def run_episodes(env: House, num_episodes: int ,policy: list = None, ): 
    total_costs_episodes = []

    num_time_steps = (env.num_years // env.time_step)+1
    rewards_all_episodes = np.zeros((num_episodes, num_time_steps))
    states_all_episodes = np.zeros((num_episodes, num_time_steps))

    # Run episodes
    for episode in range(num_episodes):
        # print("Episode:", episode + 1)
        state = env.reset()
        done = False
        time_idx = 0
        total_cost = 0
        rewards_current_episode = np.zeros(num_time_steps)
        states_current_episode = np.zeros(num_time_steps)
        while not done:
            if policy is None:
                action = env.action_space.sample()  # Random action for demonstration
            else:
                action = policy[state]
            next_state, reward, done, _ = env.step(action)
            # print(f"\tTime step: {time_idx} --> Current State: {state} | Action: {action} | Next State: {next_state} | Reward: {reward:.2f}")
            total_cost += abs(reward)
            discountfactor = 0.97**env.time
            new_reward = reward * discountfactor
            rewards_current_episode[time_idx] = (abs(np.round(reward)))
            states_current_episode[time_idx] = state
            state = next_state
            time_idx += 1

        rewards_all_episodes[episode, :] = rewards_current_episode
        states_all_episodes[episode, :] = states_current_episode
        # print("Episode finished.\n")

        total_costs_episodes.append(total_cost)

    return total_costs_episodes, rewards_all_episodes, states_all_episodes


def plot_histogram_comparisons(data1, data2):
    # Create a new figure with 1 row and 2 columns of subplots
    plt.figure(figsize=(10, 5))

    # Plot histogram for data1
    plt.hist(data1, bins=10, color='blue', alpha=0.7, label='Do-nothing policy')

    # Plot histogram for data2
    plt.hist(data2, bins=10, color='red', alpha=0.7, label='Optimal policy')

    plt.xlabel("Cost [€]")
    plt.ylabel("Frequency")
    plt.title("Histogram Comparison")
    plt.legend()
    plt.savefig('Histogram Comparison.png', dpi=300)
    # plt.show()


def plot_costs_for_policy(env: House, policy: list, rewards: np.ndarray, states: np.ndarray, plot_title: str):
    plt.close()

    actions = {0: 'DN', 1: 'R', 2: 'W', 3: 'C', 4: 'R+W', 5: 'R+F', 6: 'W+F', 7: 'All'}
    colors = {'DN': 'black', 'R': 'red', 'W': 'green', 'C': 'blue', 'R+W': 'pink', 'R+F': 'olive', 'W+F': 'purple', 'All': 'Brown' }
    # labels = {'DN': 'do nothing', 'R': 'roof', 'W': 'wall', 'C': 'cellar'}

    time_axis = np.arange(0, env.num_years+env.time_step, env.time_step)
    plt.plot(time_axis, rewards, marker='o', linestyle='-')
    for i, (xi, yi) in enumerate(zip(time_axis, rewards)):
        k = int(states[i])
        string = actions[policy[k]]
        plt.text(xi, yi, string, color=colors[string], weight='bold', ha='center', va='bottom')
 
    for i, (xi, yi) in enumerate(zip(time_axis, rewards)):
        state = int(states[i])
        value = env.state_space[state]
        letters = ['10%','20%','50%']
        num0 = int(value[0])
        num1 = letters[int(value[1])]
        num2 = letters[int(value[2])]
        num3 = letters[int(value[3])]
        string = f"{num0},R:{num1},W:{num2},F:{num3}"
        # string = str(int(states[i]))
        # string = str(value)
        plt.text(xi, yi, string,rotation = 45, weight='bold', ha='right', va='top')

    plt.ylabel("Costs [€]")
    plt.xlabel("Years")
    plt.title(f"{plot_title} policy")

    plt.grid()
    plt.savefig(f'{plot_title}.png', dpi=300)
    # plt.show()


def plot_energy_bills_for_policy(env: House, policy: list, rewards: np.ndarray, states: np.ndarray, plot_title: str):
    plt.close()

    actions = {0: 'DN', 1: 'R', 2: 'W', 3: 'C', 4: 'R+W', 5: 'R+F', 6: 'W+F', 7: 'All'}
    colors = {'DN': 'black', 'R': 'red', 'W': 'green', 'C': 'blue', 'R+W': 'pink', 'R+F': 'olive', 'W+F': 'purple', 'All': 'Brown' }
    # labels = {'DN': 'do nothing', 'R': 'roof', 'W': 'wall', 'C': 'cellar'}
    shape = np.shape(states)
    bill_per_state = np.zeros(shape=shape)
    for index,state in enumerate(states): 
        state = int(state)
        state_name = tuple(env.state_space[state])
        bill = env.energy_bills[state_name]
        bill_per_state[index] = bill

    time_axis = np.arange(0, env.num_years+env.time_step, env.time_step)
    plt.plot(time_axis, bill_per_state, marker='o', linestyle='-')
    for i, (xi, yi) in enumerate(zip(time_axis, bill_per_state)):
        k = int(states[i])
        string = actions[policy[k]]
        plt.text(xi, yi, string, color=colors[string], weight='bold', ha='center', va='bottom')

    for i, (xi, yi) in enumerate(zip(time_axis, bill_per_state)):
        state = int(states[i])
        bill = int(bill_per_state[i])
        value = env.state_space[state]
        letters = ['0','20','40']
        num0 = int(value[0])
        num1 = letters[int(value[1])]
        num2 = letters[int(value[2])]
        num3 = letters[int(value[3])]
        string = f"{num0},R:{num1},W:{num2},F:{num3}%, Bill:{bill}"
        # string = str(int(states[i]))
        # string = str(value)
        plt.text(xi, yi, string,rotation = 45, weight='bold', ha='right', va='top')

    plt.ylabel("Bills [€]")
    plt.xlabel("Years")
    plt.title(f"{plot_title} policy")

    plt.grid()
    plt.savefig(f'{plot_title}.png', dpi=300)
    # plt.show()

if __name__ == "__main__":
    # print('House')
    env = House()
    env.reset()
    print('zero')
    zero_policy = [0 for _ in range(env.observation_space.n)] #create a zero policy

    # print('Run_episodes zero')
    # # Evaluate "do-nothing" policy
    # print('Zero_policy episodes is starting')
    total_costs_zero_policy, rewards_all_episodes_zero_policy, states_all_episodes_zero_policy = run_episodes(env=env, policy=zero_policy, num_episodes=1000)  # run the zero policy

    # print('Run_episodes optimal')
    # # plot costs/policy/states for 1st episode
    for i in range(3):
        plot_costs_for_policy(env, zero_policy, rewards_all_episodes_zero_policy[i], states_all_episodes_zero_policy[i], plot_title=f'Do_nothing_{i}')
        plot_energy_bills_for_policy(env, zero_policy, rewards_all_episodes_zero_policy[i], states_all_episodes_zero_policy[i], plot_title=f'Do_nothing_Energy_bill{i}')

    # Evaluate using value iteration
    env.reset()
    # optimal_policy, optimal_value, num_iterations = value_iteration(env,continue_value_iteration=False)
    # np.save('optimal_policy.npy',optimal_policy)
    # np.save('optimal_value.npy',optimal_value)
    # print(num_iterations)
    optimal_policy = np.load('optimal_policy.npy')
    optimal_value = np.load('optimal_value.npy')
    num_iterations = 7
    print('Optimal_policy episodes  is starting')
    total_costs_value_iteration, rewards_all_episodes_value_iteration, states_all_episodes_value_iteration = run_episodes(env=env, policy=optimal_policy, num_episodes=1000)

    # # plot costs/policy/states for 1st episode
    for i in range(3):
        plot_costs_for_policy(env, optimal_policy, rewards_all_episodes_value_iteration[i], states_all_episodes_value_iteration[i], plot_title=f'Optimal_Policy_{i}')
        plot_energy_bills_for_policy(env, optimal_policy, rewards_all_episodes_value_iteration[i], states_all_episodes_value_iteration[i], plot_title=f'Optimal_Policy_Energy_bill_{i}')
    print(f"Number of iterations for optimal policy: {num_iterations}")
    plot_histogram_comparisons(total_costs_zero_policy, total_costs_value_iteration)

    #find the mean and the variance of each policy'
    #mean = policy1+policy2+policy3+...policyn / n
    #subtract the mean from each policy variance :v1= (mean - policy1)^2
    #variance = v1 + v2 +v3 +...vn / n
    mean_zero_policy = sum(total_costs_zero_policy)/ len(total_costs_zero_policy)
    mean_zero_variance = []
    for value in range(len(total_costs_zero_policy)): 
        variance = (mean_zero_policy - total_costs_zero_policy[value])**2
        mean_zero_variance.append(variance)
    mean_zero_variance = sum(mean_zero_variance)/len(mean_zero_variance)
    mean_zero_variance = np.sqrt(mean_zero_variance)

    #find mean and variance for optimal policy
    mean_optimal_policy = sum(total_costs_value_iteration)/ len(total_costs_value_iteration)
    mean_optimal_variance = []
    for value in range(len(total_costs_value_iteration)): 
        optimal_variance = (mean_optimal_policy - total_costs_value_iteration[value])**2
        mean_optimal_variance.append(optimal_variance)
    mean_optimal_variance = sum(mean_optimal_variance)/len(mean_optimal_variance)
    mean_optimal_variance = np.sqrt(mean_optimal_variance)

    print(f"The mean return of optimal policy is: {mean_optimal_policy}. The variance is:{mean_optimal_variance} ")

    print(f"The mean return of zero policy is: {mean_zero_policy}. The variance is:{mean_zero_variance} ")
    
    np.save('optimal_policy.npy',optimal_policy)
    print(optimal_policy)
