import numpy as np
import matplotlib.pyplot as plt
from house_old_fixed import House

env = House()

op = np.load('optimal_policy.npy')
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
fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)


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

plt.show()
