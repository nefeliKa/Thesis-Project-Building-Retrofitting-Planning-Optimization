



#################################################MAKING A MEAN VARIANCE FOR THE WHOLE CURVE

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

time_points = [0,5,100,1000,4000,5000]

# Original data
data = {
    "WL-EPS 1-S": [2.498, 2.498, 2.004, 1.497, 1.454, 1.452],
    "WL-EPS 1-1": [2.222, 2.222, 1.725, 1.370, 1.346, 1.322],
    "WL-EPS 2-S": [2.158, 2.158, 1.825, 1.486, 1.535, 1.535],
    "WL-EPS 2-2": [1.968, 1.968, 1.671, 1.578, 1.453, 1.450],
    "WN-EPS 1-S": [2.485, 2.485, 1.965, 1.493, 1.469, 1.424],
    "WN-EPS 1-1": [2.221, 2.221, 1.687, 1.368, 1.367, 1.338],
    "WN-EPS 2-S": [2.157, 2.157, 1.860, 1.694, 1.588, 1.575],
    "WN-EPS 2-2": [1.984, 1.984, 1.698, 1.566, 1.471, 1.472]
}

# Initialize lists to store normalized values and distinct days
normalized_values = []

# Iterate through each row
for row in data.values():
    # Calculate initial value for normalization
    initial_value = row[0]

    # Normalize each day value and append to the normalized values list
    normalized_row = [(initial_value - value) / initial_value for value in row]
    normalized_values.append(normalized_row)

# Convert the normalized values list to a NumPy array
normalized_array = np.array(normalized_values)

# Calculate the mean along the columns (axis 0)
mean_values = np.mean(normalized_array, axis=0)
print("Mean values of each column:", mean_values)

# Calculate the standard deviation for the entire normalized data
std_dev_values = np.std(normalized_array)
print("Standard deviation for the entire data:", std_dev_values)

# Convert the mean values to represent percentages
percentage_mean_values = (1 - mean_values) * 100

# Logarithmic function
def logarithmic_function(t, a, b):
    return a * np.log(b * t + 1)

# Generate time points for the new curve (years)
time_curve_years = np.linspace(0, 150, 150)

# Fit the logarithmic curve to the data
popt_logarithmic, _ = curve_fit(logarithmic_function, time_points, mean_values)

# Calculate degradation values for each year using the fitted logarithmic curve
degradation_logarithmic_years = logarithmic_function(time_curve_years * 365, *popt_logarithmic)

# Calculate upper and lower bounds by adding and subtracting standard deviation from the main curve
upper_bound_years = degradation_logarithmic_years + std_dev_values/2
lower_bound_years = degradation_logarithmic_years - std_dev_values/2

# Plot the main logarithmic curve (years) with widening upper and lower bounds
plt.figure(figsize=(10, 6))
plt.plot(time_curve_years, degradation_logarithmic_years, 'g-', label='Main Logarithmic Curve (Years)')
plt.fill_between(time_curve_years, lower_bound_years, upper_bound_years, color='gray', alpha=0.2, label='Range (±1 Standard Deviation)')
plt.xlabel('Time (years)')
plt.ylabel('Degradation')
plt.title('Main Logarithmic Curve with Widening Range (Years)')
plt.legend()
plt.grid(True)
plt.gca().invert_yaxis()
plt.show()




















###############################################APPLYING GAMMA DISTRIBUTION FOR EACH YEAR

################## make gamma distribution

## take each value of each year until the end of the curve from the mean log curve 

# take the upper and lower variance bounds of each year until the end of the curve

# make a gamma disctirbution for each year using the mean, and variance values of that year .  


from scipy.stats import gamma

# Initialize lists to store gamma distribution parameters for each year
gamma_params = []

# Iterate through each year
for year_index in range(1,len(time_curve_years)):
    # Get the mean and standard deviation of the degradation value for this year
    mean_degradation = degradation_logarithmic_years[year_index]
    std_dev_degradation = (upper_bound_years[year_index] - lower_bound_years[year_index]) / 4  # Assuming ±1 std dev corresponds to 2 std dev

    # Ensure std deviation is not zero to avoid division by zero
    std_dev_degradation = max(std_dev_degradation, 1e-6)

    # Calculate shape and scale parameters for gamma distribution
    shape = (mean_degradation **2) / (std_dev_degradation ** 2)
    scale = mean_degradation / (std_dev_degradation ** 2) 

    # Append parameters to the list
    gamma_params.append((shape, scale))

# Initialize list to store gamma distribution values for each year
gamma_distributions = []

# Generate gamma distribution for each year using its parameters
for shape, scale in gamma_params:
    gamma_distribution = gamma.rvs(a=shape, scale=scale, size=1000)  # Generating 1000 samples
    gamma_distributions.append(gamma_distribution)

import math

# Plot gamma distributions in batches
num_batches = math.ceil(len(gamma_distributions) / 15)  # Calculate the number of batches required

for batch_num in range(num_batches):
    plt.figure(figsize=(15, 10))  # Adjusting figure size for each batch
    
    # Plot up to 15 gamma distributions in this batch
    for i in range(batch_num * 15, min((batch_num + 1) * 15, len(gamma_distributions))):
        plt.subplot(3, 5, i % 15 + 1)  # Creating a subplot for each gamma distribution
        plt.hist(gamma_distributions[i], bins=30, alpha=0.5)  # Plotting the histogram for the gamma distribution
        plt.xlabel('Degradation')
        plt.ylabel('Frequency')
        plt.title(f'Gamma Distribution for Year {i}')
        plt.grid(True)
        plt.gca().invert_yaxis()
    
    plt.tight_layout()  # Adjust layout to prevent overlapping of subplots
    plt.show()



#Chose a random point in each gamma distribution using monte carlo random sampling. 

# create a new curve based on these. 
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma

# Function to generate gamma distribution parameters for each year
def generate_gamma_params(degradation_logarithmic_years, upper_bound_years, lower_bound_years):
    # Initialize an empty list to store gamma distribution parameters
    gamma_params = []
    
    # Iterate over each year (excluding the first one)
    for year_index in range(1, len(degradation_logarithmic_years)):
        # Get the mean degradation value for this year
        mean_degradation = degradation_logarithmic_years[year_index]
        
        # Calculate the standard deviation of degradation for this year
        std_dev_degradation = (upper_bound_years[year_index] - lower_bound_years[year_index]) / 4
        
        # Ensure the standard deviation is not zero to avoid division by zero
        std_dev_degradation = max(std_dev_degradation, 1e-6)
        
        # Calculate the shape parameter for the gamma distribution
        shape = (mean_degradation ** 2) / (std_dev_degradation ** 2)
        
        # Calculate the scale parameter for the gamma distribution
        scale = mean_degradation / (std_dev_degradation ** 2)
        
        # Append the shape and scale parameters to the list as a tuple
        gamma_params.append((shape, scale))
    
    # Return the list of gamma distribution parameters
    return gamma_params


# Generate gamma distribution parameters
gamma_params = generate_gamma_params(degradation_logarithmic_years, upper_bound_years, lower_bound_years)

# Generate new curves using Monte Carlo random sampling
num_curves = 100
curves = []

for _ in range(num_curves):
    curve = []
    for shape, scale in gamma_params:
        gamma_value = gamma.rvs(a=shape, scale=scale, size=1)[0]
        curve.append(gamma_value)
    curves.append(curve)

# Plot a sample of curves
num_plots = 10  # Number of curves to plot
for i in range(num_plots):
    plt.plot(curves[i], label=f'Curve {i+1}')

plt.xlabel('Years')
plt.ylabel('Conductivity ')
plt.title('Sample of Generated Curves')
plt.legend()
plt.grid(True)
plt.show()


# create categories [S1 =0-10, S2 =11-20, S3=21-30  etc]

# for each year, store all the values of conductivity into an 1-d array
##example
# year 1 =[0.2, 0.3,0.1,0.2...]


# find how many values  in the array exist into each catogory  

##example
#          S1    S2      S3 
# year 1 =[200   500    300]

# divide each array with the total sum of the values
##example
#          S1    S2      S3 
# year 1 =[200/1000   500/1000    300/1000]

 #create categories [S1 =0-10, S2 =11-20, S3=21-30  etc]

# Initialize dictionary to store conductivity values for each year
# conductivity_by_year = {}

# Get the number of columns and rows
num_columns = len(curves[0])
num_rows = len(curves)

# Create an empty array with 150 lists
conductivity_by_year = [[] for _ in range(num_columns)]

# Iterate over each list in 'curves' to populate columns
for row in curves:
    for col, value in enumerate(row):
        conductivity_by_year[col].append(value)

#
    # Define conductivity categories
categories = {'S1': (0, 10), 'S2': (11, 20), 'S3': (21, 30)}  # Add more categories as needed
    
# Initialize dictionary to store counts for each category
category_counts = {category: 0 for category in categories}


import pandas as pd 


# Assuming 'categories' and 'conductivity_by_year' are defined elsewhere

# Initialize a Pandas DataFrame
dataframe = pd.DataFrame()

# Iterate through each state and its corresponding range of values
for each_state, values_range in categories.items():
    # Iterate through each tuple containing year and conductivity value
    conductivity_idx = 0
    for year_values_array in range(len(conductivity_by_year)):
        for index in range(len(conductivity_by_year[year_values_array])):
            # Check if the conductivity value falls within the specified range
            conductivity_value = conductivity_by_year[year_values_array][index]
            if conductivity_value >= values_range[0] and conductivity_value <= values_range[1]:
                # Assign the conductivity value to the corresponding state and year in the DataFrame
                if year_values_array not in dataframe.index:
                    dataframe.loc[year_values_array] = pd.Series({each_state: conductivity_value})
                else:
                    dataframe.at[year_values_array, each_state] = conductivity_value




# Count values falling into each category
# for category, (lower_bound, upper_bound) in categories.items():    

# # Count values falling into each category
# for category, (lower_bound, upper_bound) in categories.items():
#     category_count = np.sum((year_values >= lower_bound) & (year_values <= upper_bound))
#     category_counts[category] = category_count

# Store category counts for this year
conductivity_by_year[year] = category_counts

# Calculate proportions for each category
proportions_by_year = {}

# Iterate through conductivity by year
for year, category_counts in conductivity_by_year.items():
    total_count = sum(category_counts.values())
    proportions = {category: count / total_count for category, count in category_counts.items()}
    proportions_by_year[year] = proportions

# Example of printing proportions
for year, proportions in proportions_by_year.items():
    print(f'Year {year}: {proportions}')