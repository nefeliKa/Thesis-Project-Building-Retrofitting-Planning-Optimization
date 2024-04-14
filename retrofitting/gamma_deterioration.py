import numpy as np
import matplotlib.pyplot as plt

# this script was blindly ported from Matlab to Python...
#In this script a custom gamma fucntio is used to describe the deterioration curve
#
SIMPLE_STUFF = True

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

N = 1000                                                # number of realizations

variance = std * std
b = variance / mean                                     # scale
a = (mean * mean / variance) / (np.power(T, beta))      # shape. TODO: Why is it scaled by T^b ???
# a = 10
# b = 0.2


deterioration = np.zeros((N, T + 1))
to = np.zeros(N)

for i in range(N):
    if SIMPLE_STUFF:
        to[i] = 1
        deterioration[i, 0: int(to[i])] = 0
    else:
        to[i] = np.random.randint(1, T - 1)
        deterioration[i, int(to[i])] = np.random.uniform(0, 1)

    for t in range(int(to[i]), T):
        point_from_gamma_dist = custom_gamma(a, b, t, beta)
        deterioration[i, t+1] = deterioration[i, t] + point_from_gamma_dist
        deterioration[i, t+1] = min(.9999999, float(deterioration[i, t+1]))

    plt.plot(deterioration[i, :])

D_mean = np.mean(deterioration, axis=0)
plt.plot(D_mean, linewidth=4, color="red")
plt.show()

bin_size = 10
n_bins = int(100 / bin_size)
Dlabel = deterioration.copy()

for k in range(N):
    for l in range(int(to[k]), T + 1):
        for j in range(n_bins):
            if (deterioration[k, l] < (j * bin_size / 100)) and (deterioration[k, l] >= (j - 1) * bin_size / 100):
                Dlabel[k, l] = j
                if j == 10:
                    print("bla")

trans_counts = np.zeros((n_bins, n_bins, T))

for i in range(N):
    for t in range(int(to[i]), T + 1):
        trans_counts[int(Dlabel[i, t - 1]), int(Dlabel[i, t]), t - 1] =\
            1 + trans_counts[int(Dlabel[i, t - 1]), int(Dlabel[i, t]), t - 1]

counts = np.sum(trans_counts, axis=1, keepdims=True)

p = trans_counts.copy()

for i in range(n_bins):
    for t in range(1, T+1):
        if counts[i, 0, t - 1] > 0:
            p[i, :, t - 1] = trans_counts[i, :, t - 1] / counts[i, 0, t - 1]
        else:
            p[i, :, t - 1] = 0

if SIMPLE_STUFF:
    np.save("transition_matrices_simple.npy", p)
else:
    np.save("transition_matrices_complex.npy", p)


