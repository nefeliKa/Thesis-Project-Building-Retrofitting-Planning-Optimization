from gym import Env
from house import House
from value_iteration import value_iteration


def run_episodes(env: Env, policy: list = None) -> list:
    total_costs_episodes = []

    # Run episodes
    for episode in range(5):
        print("Episode:", episode + 1)
        state = env.reset()
        done = False
        time_idx = 0
        total_cost = 0
        while not done:
            if policy is None:
                action = env.action_space.sample()  # Random action for demonstration
            else:
                action = policy[time_idx]
            next_state, reward, done, _ = env.step(action)
            print("\tCurrent State:", state, "Action:", action, "Next State:", next_state, "Reward:", reward)
            total_cost += abs(reward)
            state = next_state
            time_idx += 1
        print("Episode finished.\n")

        total_costs_episodes.append(total_cost)

    return total_costs_episodes


if __name__ == "__main__":

    env = House()

    # Evaluate random policy
    total_costs_random_policy = run_episodes(env=env, policy=None)

    # Evaluate using value iteration
    env.reset()
    optimal_policy, optimal_value = value_iteration(env)

    total_costs_value_iteration = run_episodes(env=env, policy=optimal_policy)

    print("\t\t##### Returns per episode #####\n")
    print(f"Random policy: {total_costs_random_policy}")
    print(f"Optimal policy (value iteration): {total_costs_value_iteration}")
