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
# print("Mean values of each column:", mean_values)

# Calculate the standard deviation along the columns (axis 0)
std_dev_values = np.std(normalized_array, axis=0)
# print("Standard deviation of each column:", std_dev_values)

# Convert the mean values to represent percentages
percentage_mean_values = (1 - mean_values) * 100

# Logarithmic function
def logarithmic_function(t, a, b):
    return a * np.log(b * t + 1)

# Fit the logarithmic curve to the data
popt_logarithmic, _ = curve_fit(logarithmic_function, time_points, mean_values)

# Generate time points for the curve (days)
time_curve_days = np.linspace(min(time_points), max(time_points), 1000)

# Calculate degradation values for the main logarithmic curve (days)
degradation_logarithmic_days = logarithmic_function(time_curve_days, *popt_logarithmic)

# Calculate degradation values for each year using the fitted logarithmic curve
degradation_logarithmic_years = logarithmic_function(time_curve_days, *popt_logarithmic)

popt_above = np.copy(popt_logarithmic)
popt_above[0] = popt_above[0] * 0.7
degradation_logarithmic_days_above = logarithmic_function(time_curve_days, *popt_above)

popt_below = np.copy(popt_logarithmic)
popt_below[0] = popt_below[0] * 1.1
degradation_logarithmic_days_below = logarithmic_function(time_curve_days, *popt_below)


# Plot the main logarithmic curve (days)
plt.figure(figsize=(10, 6))
plt.plot(time_curve_days, degradation_logarithmic_days, 'm-', label='Main Logarithmic Curve (Days)')
plt.fill_between(time_curve_days, degradation_logarithmic_days_below, degradation_logarithmic_days_above, color='gray', alpha=0.2, label='Range (±1 Standard Deviation)')
plt.xlabel('Time (days)')
plt.ylabel('Degradation')
plt.title('Main Logarithmic Curve with Range (Days)')
plt.legend()
plt.grid(True)
plt.gca().invert_yaxis()
plt.show()



# Logarithmic function
def logarithmic_function(t, a, b):
    return a * np.log(b * t + 1)

# Generate time points for the new curve (years)
time_curve_years = np.linspace(0, 150, 1000)

# Fit the logarithmic curve to the data
popt_logarithmic, _ = curve_fit(logarithmic_function, time_points, mean_values)


# Calculate degradation values for each year using the fitted logarithmic curve
degradation_logarithmic_years = logarithmic_function(time_curve_years * 365, *popt_logarithmic)

#Calculate variance
popt_above = np.copy(popt_logarithmic)
popt_above[0] = popt_above[0] * 0.7
degradation_logarithmic_years_above = logarithmic_function(time_curve_years * 365, *popt_above)

popt_below = np.copy(popt_logarithmic)
popt_below[0] = popt_below[0] * 1.1
degradation_logarithmic_years_below = logarithmic_function(time_curve_years * 365, *popt_below)

# Plot the main logarithmic curve (years) with widening upper and lower bounds
plt.figure(figsize=(10, 6))
plt.plot(time_curve_years, degradation_logarithmic_years, 'g-', label='Main Logarithmic Curve (Years)')

plt.fill_between(time_curve_years, degradation_logarithmic_years_below, degradation_logarithmic_years_above, color='gray', alpha=0.2, label='Range (±1 Standard Deviation)')
plt.xlabel('Time (years)')
plt.ylabel('Degradation')
plt.title('Main Logarithmic Curve with Widening Range (Years)')
plt.legend()
plt.grid(True)
plt.gca().invert_yaxis()
plt.show()