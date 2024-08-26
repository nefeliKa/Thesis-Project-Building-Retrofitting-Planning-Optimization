from gym import Env
from house_old_fixed import House
from value_iteration_old_fixed import value_iteration
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import seaborn as sns

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
    plt.figure(figsize=(12, 8))

    # Plot histogram for data1
    plt.hist(data1, bins='auto', color='blue', alpha=0.7, label='Do-nothing policy')

    # Plot histogram for data2
    plt.hist(data2, bins='auto', color='red', alpha=0.7, label='Optimal policy')

    plt.xlabel("Cost [€]")
    plt.ylabel("Frequency")
    plt.title("Histogram Comparison")
    plt.legend()
    plt.savefig('Histogram Comparison.png', dpi=300)
    # plt.show()


def plot_costs_for_policy(env: House, policy: list, rewards: np.ndarray, states: np.ndarray, plot_title: str):
    plt.close()
    plt.figure(figsize=(12, 8))
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
        letters = ['0%','20%','40%']
        num0 = int(value[0])
        num1 = letters[int(value[1])]
        num2 = letters[int(value[2])]
        num3 = letters[int(value[3])]
        string = f"{num0},R:{num1},W:{num2},F:{num3}"
        # string = str(int(states[i]))
        # string = str(value)
        plt.text(xi, yi, string,rotation = 45, weight='light', ha='right', va='top', fontsize = 10)

    plt.ylabel("Costs [€]")
    plt.xlabel("Years")
    plt.title(f"{plot_title} policy")

    plt.grid()
    plt.savefig(f'{plot_title}.png', dpi=300)
    # plt.show()


def plot_energy_bills_for_policy(env: House, policy: list, rewards: np.ndarray, states: np.ndarray, plot_title: str):
    plt.close()
    plt.figure(figsize=(12, 8))
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
        plt.text(xi, yi, string, color=colors[string], weight='light', ha='center', va='bottom')

    for i, (xi, yi) in enumerate(zip(time_axis, bill_per_state)):
        state = int(states[i])
        bill = int(bill_per_state[i])
        value = env.state_space[state]
        letters = ['0','20','40']
        num0 = int(value[0])
        num1 = letters[int(value[1])]
        num2 = letters[int(value[2])]
        num3 = letters[int(value[3])]
        string = f"{num0},R:{num1},W:{num2},F:{num3}%, kWh/m2:{bill}"
        # string = str(int(states[i]))
        # string = str(value)
        plt.text(xi, yi, string,rotation = 45, weight='light', ha='right', va='top')

    plt.ylabel("kWh per m2 [€]")
    plt.xlabel("Years")
    plt.title(f"{plot_title} policy")

    plt.grid()
    plt.savefig(f'{plot_title}.png', dpi=300)
    # plt.show()



def plot_distribution_per_time_step(data, time_step,title):
    plt.figure(figsize=(12, 8))
    
    values = data[:, time_step]

    # Plot histogram
    sns.histplot(values, kde=True, bins=20, color='blue', edgecolor='black')
    
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plot_title = plt.title(f'Distribution of Values at Time Step {time_step + 1}_{title}')
    plt.grid(True)
    plt.savefig(f'{plot_title}.png', dpi=300)
    # plt.show()


def plot_degradation(env: House, states, title):
    # Ignore the last value of each row
    plt.close()

    shape = np.shape(states)
    bill_per_state = np.zeros(shape=shape)
    for index, array in enumerate(states):
        list = [] 
        for i, state in enumerate(array): 
            state = int(state)
            state_name = tuple(env.state_space[state])
            bill = env.energy_bills[state_name]
            list.append(bill)
        ar = np.array(list)
        bill_per_state[index] = ar
    
    data = bill_per_state
    data = data[:, :-1]
    # Calculate the median and percentiles for each time step
    median_values = np.median(data, axis=0)
    percentile_25 = np.percentile(data, 25, axis=0)
    percentile_75 = np.percentile(data, 75, axis=0)

    # Calculate the error bars
    lower_error = median_values - percentile_25
    upper_error = percentile_75 - median_values

    # Plotting
    time_steps = np.arange(1, data.shape[1] + 1)

    plt.figure(figsize=(12, 8))
    
    for row in data:
        plt.plot(time_steps, row, color='#D3D3D3', linewidth=0.1)  # Light gray color

    # Plot the mean line with error bars
    plt.errorbar(time_steps, median_values, yerr=[lower_error, upper_error], fmt='o', capsize=5, color='blue', label='Median with IQR')
    plt.plot(time_steps, median_values, color='blue', linestyle='-', linewidth=2, label='Median Value')

    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plot_title = plt.title(f'Degradation Over Time with Individual and Median Trends_{title}')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{plot_title}.png', dpi=300)
    # plt.show()
    
    # Plot distribution for each time step
    for time_step in range(data.shape[1]):
        plot_distribution_per_time_step(data, time_step,title)


def plot_actions(env,op): 

    states = np.load('state_space.npy')
    dictionary = {}
    for year in range(0, env.num_years+env.time_step, env.time_step):
        array = np.where(states[:, 0] == year)[0]  # find the states where the first value equals the year
        ind1 = np.max(array) 
        ind2 = np.min(array)
        dictionary[year] = op[ind2:ind1]  # store actions for each year

    # Count the total occurrences of each action for each year
    total_actions_per_year = {year: len(dictionary[year]) for year in dictionary}

    # Count the occurrences of each action for each year
    action_counts = {year: {action: np.sum(dictionary[year] == action) for action in set(dictionary[year])} for year in dictionary}

    # Initialize an empty dictionary to store the action lists
    actions = ['Do_nothing','Change_Roof','Change_Facade','Change_Groundfloor','Change Roof and Facade','Change Roof and Groundfloor','Change Facade and Groundfloor', 'Change All']
    action_dictionary = {f'Action_{action}:{actions[action]}': [] for action in range(8)}

    # Iterate over years and actions to populate the action_dictionary with percentages
    for year in action_counts:
        for action in range(8):
            percentage = round((action_counts[year].get(action, 0) / total_actions_per_year[year]) * 100,1)
            action_dictionary[f'Action_{action}:{actions[action]}'].append(percentage)  # Append the percentage of the action

    # Convert the lists to tuples
    action_dictionary = {key: tuple(values) for key, values in action_dictionary.items()}

    # Print or use action_dictionary as needed
    # print(action_dictionary)

    years = tuple(dictionary.keys())
    x = np.arange(len(years))  # the label locations
    width = 0.1  # the width of the bars
    multiplier = -3

    # fig, ax = plt.subplots(layout='constrained')
    # Change the figsize parameter to adjust the size of the figure
    fig, ax = plt.subplots(figsize=(12, 8), constrained_layout=True)


    for attribute, measurement in action_dictionary.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=attribute)
        ax.bar_label(rects, padding=3)
        multiplier += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Action Occurrence Percentage')
    ax.set_title('Action Occurrence Percentage per year')
    ax.set_xticks(x + width, years)
    ax.legend(loc='upper left', ncols=3)
    ax.set_ylim(0, 200)  # Limit y-axis to percentages (0-100)
    plt.savefig('plotted_actions',dpi=300)

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
    plot_degradation(env, states=states_all_episodes_zero_policy,title= 'zero policy')

    # Evaluate using value iteration
    env.reset()
    optimal_policy, optimal_value, num_iterations = value_iteration(env,continue_value_iteration=False)
    np.save('optimal_policy.npy',optimal_policy)
    np.save('optimal_value.npy',optimal_value)
    print(num_iterations)
    # optimal_policy = np.load('optimal_policy.npy')
    # optimal_value = np.load('optimal_value.npy')
    # num_iterations = 13
    # print('Optimal_policy episodes  is starting')
    total_costs_value_iteration, rewards_all_episodes_value_iteration, states_all_episodes_value_iteration = run_episodes(env=env, policy=optimal_policy, num_episodes=1000)

    # # plot costs/policy/states for 1st episode
    for i in range(3):
        plot_costs_for_policy(env, optimal_policy, rewards_all_episodes_value_iteration[i], states_all_episodes_value_iteration[i], plot_title=f'Optimal_Policy_{i}')
        plot_energy_bills_for_policy(env, optimal_policy, rewards_all_episodes_value_iteration[i], states_all_episodes_value_iteration[i], plot_title=f'Optimal_Policy_Energy_bill_{i}')
    print(f"Number of iterations for optimal policy: {num_iterations}")
    plot_histogram_comparisons(total_costs_zero_policy, total_costs_value_iteration)
    # plot_actions(env,zero_policy,'zero policy')
    c= plot_actions(env=env,op=optimal_policy)
    title ='optimal policy'
    plot_degradation(env, states=states_all_episodes_value_iteration,title= 'optimal policy')
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
