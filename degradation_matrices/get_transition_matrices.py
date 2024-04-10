import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import gamma


def normalize_data(data):
    normalized_values = []

    # Iterate through each row
    for row in data.values():
        # Calculate initial value for normalization
        initial_value = row[0]

        # Normalize each day value and append to the normalized values list
        normalized_row = [(initial_value - value) / initial_value for value in row]
        normalized_values.append(normalized_row)

    return normalized_values

def get_mean_data(data):
    return np.mean(list(data.values()), axis=0)

def plot_data(time, data, normalized_data, mean_data):
    # Plotting original data
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.title('Original Data')
    for values in data:
        y_data = values  # Original data
        plt.plot(time, y_data)
    plt.xlabel('Days')
    plt.ylabel('Deterioration ratio (%)')

    # Plotting normalized data
    plt.subplot(1, 3, 2)
    plt.title('Normalized Data')
    for values in normalized_data:
        y_data = values  # Normalized data
        plt.plot(time, y_data)
    plt.xlabel('Days')
    plt.ylabel('Normalized Deterioration ratio')
    plt.gca().invert_yaxis()

    # Plotting mean curves
    plt.subplot(1, 3, 3)
    plt.title('Mean Curves')
    plt.plot(time, mean_data, marker='o', linestyle='-', color='red', label='Mean Curve (EPS)')
    plt.xlabel('Days')
    plt.ylabel('Normalized Deterioration ratio')
    plt.legend()

    plt.tight_layout()
    plt.show()

def fit_data(time_points, normalized_data, do_plot=False):
    mean_norm_data = np.mean(np.array(normalized_data), axis=0)

    def logarithmic_function(t, a, b):
        return a * np.log(b * t + 1)
    # Generate time points for the new curve (years)
    time_curve_years = np.linspace(0, 150, 150)

    # Fit the logarithmic curve to the data
    popt_logarithmic, _ = curve_fit(logarithmic_function, time_points, mean_norm_data)
    # Calculate degradation values for each year using the fitted logarithmic curve
    fitted_curve = logarithmic_function(time_curve_years * 365, *popt_logarithmic)


    # Calculate the standard deviation for the entire normalized data
    std_norm_data = np.std(np.array(normalized_data))
    upper_bound_years = fitted_curve + std_norm_data/2
    lower_bound_years = fitted_curve - std_norm_data/2

    if do_plot:
        plt.figure(figsize=(10, 6))
        plt.plot(np.linspace(0, 150, 150), fitted_curve, 'g-', label='Main Logarithmic Curve (Years)')
        plt.fill_between(np.linspace(0, 150, 150), lower_bound_years, upper_bound_years, color='gray', alpha=0.2, label='Range (±1 Standard Deviation)')
        plt.xlabel('Time (years)')
        plt.ylabel('Degradation')
        plt.title('Main Logarithmic Curve with Widening Range (Years)')
        plt.legend()
        plt.grid(True)
        plt.gca().invert_yaxis()
        plt.show()

    return fitted_curve, std_norm_data, upper_bound_years, lower_bound_years

# Function to generate gamma distribution parameters for each year
def generate_gamma_params(fitted_curve, upper_bound_years, lower_bound_years):
    # Initialize an empty list to store gamma distribution parameters
    gamma_params = []
    
    # Iterate over each year (excluding the first one)
    for year_index in range(1, len(fitted_curve)):
        # Get the mean degradation value for this year
        mean_degradation = fitted_curve[year_index]
        
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

def monte_carlo_sampling(gamma_params, num_curves=100):
    # Generate new curves using Monte Carlo random sampling
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


if __name__=="__main__":
    # Original data
    time_points = [0,5,100,1000,4000,5000]
    data_dict = {
        "WL-EPS 1-S": [2.498, 2.498, 2.004, 1.497, 1.454, 1.452],
        "WL-EPS 1-1": [2.222, 2.222, 1.725, 1.370, 1.346, 1.322],
        "WL-EPS 2-S": [2.158, 2.158, 1.825, 1.486, 1.535, 1.535],
        "WL-EPS 2-2": [1.968, 1.968, 1.671, 1.578, 1.453, 1.450],
        "WN-EPS 1-S": [2.485, 2.485, 1.965, 1.493, 1.469, 1.424],
        "WN-EPS 1-1": [2.221, 2.221, 1.687, 1.368, 1.367, 1.338],
        "WN-EPS 2-S": [2.157, 2.157, 1.860, 1.694, 1.588, 1.575],
        "WN-EPS 2-2": [1.984, 1.984, 1.698, 1.566, 1.471, 1.472]
    }

    mean_orig_data = get_mean_data(data_dict)
    normalized_data = normalize_data(data_dict)

    # plot_data(time=time_points, normalized_data=normalized_data, data=data_dict.values(), mean_data=mean_orig_data)

    fitted_curve, std_norm_data, upper_bounds, lower_bounds = fit_data(time_points=time_points, normalized_data=np.array(normalized_data), do_plot=False)

    gamma_params = generate_gamma_params(fitted_curve=fitted_curve,
                                         upper_bound_years=upper_bounds,
                                         lower_bound_years=lower_bounds)

    monte_carlo_sampling(gamma_params=gamma_params)

