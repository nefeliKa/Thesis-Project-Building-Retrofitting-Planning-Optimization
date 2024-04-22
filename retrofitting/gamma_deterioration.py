import numpy as np
import matplotlib.pyplot as plt

# this script was blindly ported from Matlab to Python...
#In this script a custom gamma fucntio is used to describe the deterioration curve

def matrices_gen(SIMPLE_STUFF:bool, N:int, T:int, do_plot:bool): 
    # SIMPLE_STUFF = True
    # do_plot = False
    def custom_gamma(a, b, t, beta):
        # TODO: understand why the shape, scale are selected like this. Maybe more info on the paper??
        x = np.random.gamma(shape=a * t**beta - a * (t - 1)**beta, scale = b)
        return x


    mean = 0.35                                             # the mean descibes the fianl degradation that the material 
                                                            # will have at the end of the time period  T
    beta = 0.2                                              # beta descibes the curvature and direction of the curve. 
                                                            #For a more steep curve beta shoudl be lower than 1.                                                                   
    std = 0.15 * mean
    # T = 60                                                  # number of years 
    # N = 1000                                                # number of realizations
    variance = std * std
    b = variance / mean                                     # scale
    a = (mean * mean / variance) / (np.power(T, beta))      # shape. TODO: Why is it scaled by T^b ???
    # a = 10
    # b = 0.2

    deterioration = np.zeros((N, T + 1))
    to = np.zeros(N) # time observed. For simple_stuff, the time is 1

    for iteration in range(N):
        if SIMPLE_STUFF:                                    #choose if you want the complex matrix or not
            to[iteration] = 1                               #when the matrix is simple, the beginning time is at time 1
            deterioration[iteration, 0: int(to[iteration])] = 0   
        else:
            to[iteration] = np.random.randint(1, T-1)
            deterioration[iteration, int(to[iteration])] = np.random.uniform(0, 1)  #TODO make sure the curve here follows the gamma distribution

        for time in range(int(to[iteration]), T):
            point_from_gamma_dist = custom_gamma(a, b, time, beta) #take a point that follows the gamma distribution curve
            deterioration[iteration, time +1] = deterioration[iteration, time] + point_from_gamma_dist
            deterioration[iteration, time] = min(.9999999, float(deterioration[iteration, time + 1]))

        plt.plot(deterioration[iteration, :])   # make an array of 1000 arrays. Each array contains the values of the line for the time t

    if do_plot:
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
    #Find the number of values in a category in time t 
    #Find the number of values in all the categories in the   

    list1 = []
    for t in range(T+1):
        matrix = np.zeros((3,3))
        list1.append(matrix)
    for n in range(N): 
        for t in range(T+1): 
            if t<= T-1:
                current = int(Dlabel[n,t])
                future = int(Dlabel[n,t + 1])
                list1[t][current][future]+=1

    array = np.array(list1)

    for t in range(T+1):
        for i in range(3):  # Assuming your matrix size is (3, 3)
            s = int(np.sum(array[t][i]))  # Calculate the sum of the row
            if s==0:
                if i ==0:
                    array[t][i] = np.array([0. , 1. , 0.])
                else: 
                    array[t][i] = np.array([0. , 0. , 1.])
                s = int(np.sum(array[t][i]))
            array[t][i] = np.divide(array[t][i],s)  # Divide values by the sum 

    # checker = []
    # for t in range(T+1):
    #     for i in range(3):  # Assuming your matrix size is (3, 3)
    #         s = int(np.sum(array[t][i]))  # Calculate the sum of the row
    #         if s!= 1: 
    #             checker.append[t]

    # print(checker)
    # print('bl')
    # #Save the new transition probabilities array
    np.save("transition_matrices_trial.npy", array)

    return array


# if __name__=="__main__":

#     p = matrices_gen(SIMPLE_STUFF = True,N= 1000000 ,T = 150, do_plot = False)



