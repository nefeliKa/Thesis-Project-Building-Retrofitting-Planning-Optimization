from gym import Env
from house import House


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

    # Evaluate random policy
    run_episodes(env=House(), policy=None)
