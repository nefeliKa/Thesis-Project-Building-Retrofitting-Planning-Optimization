import matplotlib.pyplot as plt
import numpy as np

op = np.load('optimal_policy.npy')
states = np.load('state_space.npy')

dictionary = {}
for year in range(0, 20, 5):
    array = np.where(states[:, 0] == year)[0]  # find the states where the first value equals the year
    ind1 = np.max(array)
    ind2 = np.min(array)
    dictionary[year] = op[ind2:ind1]  # store actions for each year

# Count the occurrences of each action for each year
action_counts = {year: {action: np.sum(dictionary[year] == action) for action in set(dictionary[year])} for year in dictionary}

# Extract unique actions
unique_actions = np.unique(op)

# Plotting
num_years = len(action_counts)
num_actions = len(unique_actions)
bar_width = 0.8 / num_actions  # Adjust bar width based on the number of actions
colors = plt.cm.tab10(np.arange(num_actions))  # Generate colors for actions

plt.figure(figsize=(12, 6))

for i, (year, counts) in enumerate(action_counts.items()):
    x_values = np.arange(num_actions) + i * (bar_width * num_actions + 0.1)  # Adjust x-values for each year
    for j, action in enumerate(unique_actions):
        height = counts.get(action, 0)
        plt.bar(x_values[j], height, bar_width, label=f'Year {year}, Action {action}', color=colors[j])

plt.xlabel('Years')
plt.ylabel('Frequency')
plt.title('Frequency of Actions Recommended by Policy for Each Year')
plt.xticks(np.arange(num_years), action_counts.keys())  # Set x-ticks to represent years
plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1))
plt.tight_layout()
plt.show()