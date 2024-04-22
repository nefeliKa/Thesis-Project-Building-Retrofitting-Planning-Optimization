import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import gamma
import pandas as pd 

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

def get_original_std_dev(data): 

    result_dict = {}

    list_length = len(next(iter(data.values())))

    # Iterate over the range of list length
    for i in range(list_length):
        # Initialize a new list for each index
        values_at_index = []
        # Iterate over the dictionary values and append the value at the current index
        for key, value_list in data.items():
            values_at_index.append(value_list[i])

        # Calculate the standard deviation of the values at the current index
        std_deviation = np.std(values_at_index)
        # Store the standard deviation in the result dictionary
        result_dict[f'std_deviation_{i+1}'] = std_deviation
        
    return result_dict


def get_normalized_std_dev(data): 
    # Initialize a new dictionary to store the standard deviations
    result_dict = {}

    # Get the length of the inner lists (assuming all lists have the same length)
    list_length = len(data[0])

    # Iterate over the range of list length
    for i in range(list_length):
        # Initialize a new list for each index
        values_at_index = []
        # Iterate over the lists in the data and append the value at the current index
        for sublist in data:
            values_at_index.append(sublist[i])
        # Calculate the standard deviation of the values at the current index
        std_deviation = np.std(values_at_index)
        # Store the standard deviation in the result dictionary
        result_dict[f'std_deviation_{i+1}'] = std_deviation

    return result_dict



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

def fit_data(time_points, normalized_data, lifespan ,do_plot=False ):
    mean_norm_data = np.mean(np.array(normalized_data), axis=0)

    def logarithmic_function(t, a, b):
        return a * np.log(b * t + 1)
    
    # def gamma_function(x, a, b, c):
    #     return a * np.exp(-b * x) * x**(c - 1)
    
    # Generate time points for the new curve (years)
    time_curve_years = np.linspace(start =0, stop = lifespan, num =lifespan)

    # # Fit the logarithmic curve to the data
    popt_logarithmic, _ = curve_fit(logarithmic_function, time_points, mean_norm_data)
    # Calculate degradation values for each year using the fitted logarithmic curve
    fitted_curve = logarithmic_function(time_curve_years * 365, *popt_logarithmic)

    # Fit the logarithmic curve to the data
    # popt_gamma, _ = curve_fit(gamma_function, time_points, mean_norm_data, p0=[1, 1, 1])
    # Calculate degradation values for each year using the fitted logarithmic curve
    # fitted_curve = gamma_function(time_curve_years * 365, *popt_gamma)

    # Calculate the standard deviation for the entire normalized data
    std_norm_data = np.std(np.array(normalized_data))
    upper_bound_years = fitted_curve + std_norm_data/2
    lower_bound_years = fitted_curve - std_norm_data/2

    #Make time points into years
    days2years = [value/365 for value in time_points] 

    if do_plot:
        plt.figure(figsize=(10, 6))
        plt.scatter(days2years,mean_norm_data, color = 'red')
        plt.plot(time_curve_years, fitted_curve, 'g-', label='Main Logarithmic Curve (Years)')
        plt.fill_between(time_curve_years, lower_bound_years, upper_bound_years, color='gray', alpha=0.2, label='Range (±1 Standard Deviation)')
        plt.xlabel('Time (years)')
        plt.ylabel('Degradation')
        plt.title('Main Logarithmic Curve with Widening Range (Years)')
        plt.legend()
        plt.grid(True)
        # plt.gca().invert_yaxis()
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
        std_dev_degradation = abs(upper_bound_years[year_index]) - abs(lower_bound_years[year_index])
        
        # # Ensure the standard deviation is not zero to avoid division by zero
        std_dev_degradation = max(std_dev_degradation, 1e-6)
        
        # Calculate the shape parameter for the gamma distribution
        shape = (mean_degradation ** 2) / (std_dev_degradation ** 2)
        
        # Calculate the scale parameter for the gamma distribution
        scale = mean_degradation / (std_dev_degradation ** 2)
        
        # Append the shape and scale parameters to the list as a tuple
        gamma_params.append((shape, scale))
    
    # Return the list of gamma distribution parameters
    return gamma_params

def sampling2(fitted_curve,building_lifespan, do_plot):
    crv = []
    for year_index in range(len(fitted_curve)):
        if year_index == 0:
            crv.append(fitted_curve[year_index])  # Append the first value from the fitted curve
        else:
            mean = crv[-1]  # Set the mean as the previous point's value
            beta = 0.2  # Adjust beta as needed
            std = 0.15 * mean  # Adjust standard deviation as needed
            variance = std * std
            scale = variance / mean
            shape = (mean * mean / variance) / (np.power(building_lifespan, beta))
            year_value = gamma.rvs(a=shape, scale=scale, size=1)[0]
            crv.append(max(mean, year_value))  # Ensure the sampled point is equal to or greater than the previous value

    if do_plot:
        plt.plot(crv)
        plt.xlabel('Time (years)')
        plt.ylabel('Degradation')
        plt.title('Generated Degradation over Time')
        plt.show()

    return crv




def monte_carlo_sampling(gamma_params, num_curves=2, do_plot=True):
    # Generate new curves using Monte Carlo random sampling
    curves = []

    for _ in range(num_curves):
        curve = []
        for shape, scale in gamma_params:
            gamma_value = gamma.rvs(a=shape, scale=scale, size=1)[0]
            curve.append(gamma_value)
        curves.append(curve)

    # Plot a sample of curves
    if do_plot:
        num_plots = 2  # Number of curves to plot
        for i in range(num_plots):
            plt.plot(curves[i], label=f'Curve {i+1}')

        plt.xlabel('Years')
        plt.ylabel('Conductivity ')
        plt.title('Sample of Generated Curves')
        plt.legend()
        plt.grid(True)
        plt.show()

    return curves

# Function to count total number of values in a dictionary
def count_values(d):
    total = 0
    for dict in d.values():
        for values in dict: 
            num = len(dict[values]) 
            total += num
    return total

def make_matrices(curves, categories): 

    #Get all the values of a year
    # Get the number of columns and rows
    num_columns = len(curves[0])
    num_rows = len(curves)

    # Initialize a Pandas DataFrame to store the data
    dictionary = {}

    # Create an empty array with 150 lists
    conductivity_by_year = [[] for _ in range(num_columns)]

    # Iterate over each list in 'curves' to populate columns
    for row in curves:
        for col, value in enumerate(row):
            conductivity_by_year[col].append(value)

    cond = {}
    for row,values in enumerate(conductivity_by_year): 
        cond[row] = values

    # for year, value in cond.items(): 
    #     dictionary[year] = {'S1': [], 'S2':[], 'S3':[]}
    #     for datapoint in range(len(value)):

    #         print('bla')


#   PROBLEM: because the values appear too big, I can't separate them in the correct classes. 

    for year, year_data in enumerate(conductivity_by_year):
        dictionary[year] = {'S1': [], 'S2':[], 'S3':[]}
        for value in year_data:
            for category,ranges in categories.items():
                if value >= ranges[0] and value <= ranges[1]:
                    dictionary[year][category].append(value)

    # #check the amount of values in the curves list
    # # Find the total number of values in all sublists
    # total_values = sum(len(sublist) for sublist in curves)
    # print(total_values)

    # # Initialize total values count
    # total_values2 = 0
    # for sublist in conductivity_by_year:
    #     total_values2 += len(sublist)

    # # Find the minimum and maximum values in the list of sublists
    # min_value = min(min(sublist) for sublist in conductivity_by_year)
    # max_value = max(max(sublist) for sublist in conductivity_by_year)
    # # Call the function to count total values
    # total_values3 = count_values(dictionary)
    # # print(total_values3)
    
    counted_data = {}
    #Count the amount of data in each disctionary key 
    for  year, values in dictionary.items():
        counted_data[year] = {'S1': [], 'S2':[], 'S3':[]}
        total_counts = 0
        for category,list in values.items():
            list_lenght = len(list) 
            total_counts += list_lenght
            counted_data[year][category] = list_lenght
        for key in counted_data[year]:
            if counted_data != 0:  
                counted_data[year][key] = counted_data[year][key]/ total_counts
    return counted_data

















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
    std_original_data = get_original_std_dev(data_dict)
    std_normalized_data = get_normalized_std_dev(normalized_data)
    building_lifespan = 20
    # plot_data(time=time_points, normalized_data=normalized_data, data=data_dict.values(), mean_data=mean_orig_data)

    fitted_curve, std_norm_data, upper_bounds, lower_bounds = fit_data(time_points=time_points, normalized_data=np.array(normalized_data),lifespan = building_lifespan ,do_plot=True)
    
    gamma_params = generate_gamma_params(fitted_curve=fitted_curve,
                                         upper_bound_years=upper_bounds,
                                         lower_bound_years=lower_bounds )
    

    mean = 0.35                                              # the mean descibes the fianl degradation that the material 
                                                        # will have at the end of the time period  T
    beta = 0.2                                              # beta descibes the curvature and direction of the curve. 
                                                            #For a more steep curve beta shoudl be lower than 1.                                                                   
    std = 0.15 * mean

    trial = sampling2(fitted_curve=fitted_curve, building_lifespan =building_lifespan, do_plot=True)


    curves =  monte_carlo_sampling(gamma_params=gamma_params, do_plot=True)
    

       # Define conductivity categories
    categories = {'S1': (0, 10), 'S2': (11, 30), 'S3': (31, 100)}  # Add more categories as needed

    matrices = make_matrices(curves=curves,categories=categories)
    print(matrices)
