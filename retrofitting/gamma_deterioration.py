import numpy as np
import matplotlib.pyplot as plt

# this script was blindly ported from Matlab to Python...
#In this script a custom gamma fucntio is used to describe the deterioration curve
#


def custom_gamma(a, b, t, beta):
    # TODO: understand why the shape, scale are selected like this. Maybe more info on the paper??
    x = np.random.gamma(shape=a * t**beta - a * (t - 1)**beta, scale = b)
    return x


mean = 0.35                                              # the mean descibes the fianl degradation that the material 
                                                        # will have at the end of the time period  T
beta = 0.2                                              # beta descibes the curvature and direction of the curve. 
                                                        #For a more steep curve beta shoudl be lower than 1.                                                                   
std = 0.15 * mean
T = 150                                                 # number of years 

no_realisations = 1000                                                # number of realizations

variance = std * std
b = variance / mean                                     # scale
a = (mean * mean / variance) / (np.power(T, beta))      # shape. TODO: Why is it scaled by T^b ???
# a = 10
# b = 0.2


deterioration = np.zeros((no_realisations, T + 1))
to = np.zeros((1, no_realisations))                     #an empty array with shape (1,1000)

for i in range(no_realisations):        
    to[0][i] = 1                                        #populate all array with values of 1
    deterioration[i, 0] = 0

    for t in range(1, T):
        point_from_gamma_dist = custom_gamma(a, b, t, beta) # for time t in lifespan, get a random point from the gamma distribution  
        deterioration[i, t+1] = deterioration[i, t] + point_from_gamma_dist # get the next deterioration point by adding the current deterioration with the random point that was generated 
        deterioration[i, t+1] = min(.9999999, float(deterioration[i, t+1]))  # if the value is greater that 0.999... it will become 0.999. No values can b better that 1
 
    plt.plot(deterioration[i, :])

D_mean = np.mean(deterioration, axis=0)             
plt.plot(D_mean, linewidth=4, color="red")
plt.show()

bin_size = 10                           # category changes when the deterioration drops by 'bin_size'
n_bins = int(100 / bin_size)            # calculates the number of different states 
Dlabel = deterioration.copy()           #?

# Label the category that each point of each curve is at the time t
for realisation in range(no_realisations):
    for time in range(int(to[0][realisation]), T+1):        # for range in 'to' array, ( column index [realisation] and time T)
        for state in range(n_bins + 1):                     #for state in states range
            if (deterioration[realisation, time] < (state * bin_size / 100)) and (deterioration[realisation, time] >= (state - 1) * bin_size / 100):
                Dlabel[realisation, time] = state           # if the values of detariotation fit the state size put them in the Dlabel

trans_counts = np.zeros((n_bins, n_bins, T))                #   make a 3D array 10x10x150

# Counts how many times each label appears at time t
for realisation in range(no_realisations):                            
    for time in range(int(to[0][realisation]), T + 1):
        trans_counts[int(Dlabel[realisation, time - 1]), int(Dlabel[realisation, time]), time - 1] =\
            1 + trans_counts[int(Dlabel[realisation, time - 1]), int(Dlabel[realisation, time]), time - 1]

counts = np.sum(trans_counts, axis=1, keepdims=True)

p = trans_counts.copy()

# translates that into probabilities by dividing by the total amount of counts
for i in range(n_bins):
    for t in range(1, T+1):
        if counts[i, 0, t - 1]>0 :
            p[i, :, t - 1] = trans_counts[i, :, t - 1] / counts[i, 0, t - 1]
        else: 
            p[i, :, t - 1] = 0

np.save("transition_matrices.npy",p)
watchpoint = p[:,:,0]
watchpoint2 = trans_counts[:,:,149]

print(watchpoint)

print(trans_counts[:,:,0])

