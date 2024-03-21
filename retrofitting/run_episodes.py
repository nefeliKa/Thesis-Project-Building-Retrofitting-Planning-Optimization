from gym import Env
from house import House
from value_iteration import value_iteration
import matplotlib.pyplot as plt

#TODO 
#check how many episodes I have to run it to have a good estimate
#check the results for different policies
#run 10000 episodes to evaluate policy because there is a stochasticisity 
#next time: priority difference of deterioration rate over time 
#next time : do nothing - evaluate the rewards according to that 


def run_episodes(env: Env, policy: list = None) -> list:
    total_costs_episodes = []

    # Run episodes
    for episode in range(100):
        print("Episode:", episode + 1)
        state = env.reset()
        done = False
        time_idx = 0
        total_cost = 0
        while not done:
            if policy is None:
                action = env.action_space.sample()  # Random action for demonstration
            else:
                action = policy[state]
            next_state, reward, done, _ = env.step(action)
            print(f"\tTime step: {time_idx} --> Current State: {state} | Action: {action} | Next State: {next_state} | Reward: {reward:.2f}")
            total_cost += abs(reward)
            state = next_state
            time_idx += 1
        print("Episode finished.\n")

        total_costs_episodes.append(total_cost)

    return total_costs_episodes


def get_histogram(data1, data2):
    # Create a new figure with 1 row and 2 columns of subplots
    plt.figure(figsize=(10, 5))

    # Plot histogram for data1
    plt.hist(data1, bins='auto', color='blue', alpha=0.7, label='Do-nothing policy')

    # Plot histogram for data2
    plt.hist(data2, bins='auto', color='red', alpha=0.7, label='Optimal policy')

    plt.xlabel('Cost [€]')
    plt.ylabel('Frequency')
    plt.title('Histogram Comparison')
    plt.legend()

    plt.show()


if __name__ == "__main__":

    env = House()
    zero_policy = [0 for _ in range(env.observation_space.n)] #create a zero policy
    # optimal_policy, optimal_value = value_iteration(env)

    # Evaluate random policy
    total_costs_zero_policy = run_episodes(env=env, policy=zero_policy)# run the zero policy
    #total_costs_random_policy = run_episodes(env=env, policy=None) #run a random policy

    # Evaluate using value iteration
    env.reset()
    optimal_policy, optimal_value, num_iterations = value_iteration(env)
    total_costs_value_iteration = run_episodes(env=env, policy=optimal_policy)

    print(f"Number of iterations for optimal policy: {num_iterations}")
    get_histogram(total_costs_zero_policy, total_costs_value_iteration)

