import numpy as np
import matplotlib.pyplot as plt
'''
With this gamma deterioration I  creaate probabilities for certain time steps. For example, I get probabilites for each n years and not for each year. 
However this creates the problem that 
1) you have to calculate the probabilities each time 
2) the script is not really adjusted well and you might get nan or sum_zero rows 
'''
def matrices_gen(SIMPLE_STUFF,N,T,do_plot): 
    # this script was blindly ported from Matlab to Python...
    #In this script a custom gamma fucntio is used to describe the deterioration curve
    #
    # SIMPLE_STUFF = False

    def custom_gamma(a, b, t, beta):
        # TODO: understand why the shape, scale are selected like this. Maybe more info on the paper??
        x = np.random.gamma(shape=a * t**beta - a * (t - 1)**beta, scale = b)
        return x


    mean = 0.35                                              # the mean descibes the fianl degradation that the material 
                                                            # will have at the end of the time period  T
    beta = 0.2                                              # beta descibes the curvature and direction of the curve. 
                                                            #For a more steep curve beta shoudl be lower than 1.                                                                   
    std = 0.15 * mean
    # T = 60                                                 # number of years 

    # N = 1000                                                # number of realizations

    variance = std * std
    b = variance / mean                                     # scale
    a = (mean * mean / variance) / (np.power(T, beta))      # shape. TODO: Why is it scaled by T^b ???
    # a = 10
    # b = 0.2


    deterioration = np.zeros((N, T + 1))
    to = np.zeros(N)

    for i in range(N):
        if SIMPLE_STUFF: #choose if you want the complex matrix or not
            to[i] = 1
            deterioration[i, 0: int(to[i])] = 0
        else:
            to[i] = np.random.randint(1, T - 1)
            deterioration[i, int(to[i])] = np.random.uniform(0, 1)  #TODO make sure the curve here follows the gamma distribution

        for t in range(int(to[i]), T):
            point_from_gamma_dist = custom_gamma(a, b, t, beta)
            deterioration[i, t+1] = deterioration[i, t] + point_from_gamma_dist
            deterioration[i, t+1] = min(.9999999, float(deterioration[i, t+1]))

        plt.plot(deterioration[i, :])   # make an array of 1000 arrays. Each array contains the values of the line for the time t
    if do_plot :
        D_mean = np.mean(deterioration, axis=0)
        plt.plot(D_mean, linewidth=4, color="red")
        plt.show()


    # Define custom categories
    categories = [10, 20, 100]

    # Categorize deterioration values into custom categories
    Dlabel = deterioration.copy()
    for k in range(N):
        for l in range(int(to[k]), T + 1):
            for j, threshold in enumerate(categories):
                if deterioration[k, l] <= threshold / 100:
                    Dlabel[k, l] = j
                    break


    # Calculate transition counts between categories over time
    n_bins = len(categories) + 1
    trans_counts = np.zeros((n_bins, n_bins, T))

    for i in range(N):
        for t in range(int(to[i]), T + 1):
            trans_counts[int(Dlabel[i, t - 1]), int(Dlabel[i, t]), t - 1] += 1

    # Calculate transition probabilities
    counts = np.sum(trans_counts, axis=1, keepdims=True)
    p = np.zeros_like(trans_counts)

    for i in range(n_bins):
        for t in range(1, T + 1):
            if counts[i, 0, t - 1] > 0:
                p[i, :, t - 1] = trans_counts[i, :, t - 1] / counts[i, 0, t - 1]


    # Define the value of n (e.g., every 5th year)
    # # n = 1

    # # Calculate the number of time steps in the new probability array
    # T_new = T // n

    # # Create a new array to store transition probabilities for every nth year
    # p_n = np.zeros((n_bins-1, n_bins - 1, T_new))

    # # Iterate over categories and time steps
    # for i in range(n_bins):
    #     for t in range(1, T_new + 1):
    #         # Calculate the starting index of the original array for this nth year
    #         start_index = (t - 1) * n
            
    #         # Calculate the ending index of the original array for this nth year
    #         end_index = t * n if t < T_new else T
            
    #         # Sum the transition counts for this nth year
    #         sum_counts = np.sum(trans_counts[i, :, start_index:end_index], axis=1)
            
    #         # Calculate transition probabilities for this nth year
    #         if np.sum(sum_counts) > 0:
    #             p_n[i, :, t - 1] = sum_counts[:-1] / np.sum(sum_counts[:-1])  # Exclude the last category

    # Save the new transition probabilities array
    # np.save("transition_matrices_n.npy", p_n)

    # Save transition matrices
    if SIMPLE_STUFF:
        np.save("transition_matrices_simple.npy", p)
    else:
        np.save("transition_matrices_complex.npy", p)
    return p

if __name__=="__main__":

    p = matrices_gen(SIMPLE_STUFF = True,N= 1000000 ,T = 150, do_plot = False)
