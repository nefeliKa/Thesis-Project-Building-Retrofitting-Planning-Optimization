import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
# from house_old_fixed import House
# this script was blindly ported from Matlab to Python...
#In this script a custom gamma fucntio is used to describe the deterioration curve


def matrices_gen(SIMPLE_STUFF:bool, N:int, T:int, do_plot:bool,step_size:int,save_probabilities:bool ): 
    # SIMPLE_STUFF = True
    # do_plot = False
    def custom_gamma(a, b, t, beta):
        # TODO: understand why the shape, scale are selected like this. Maybe more info on the paper??
        x = np.random.gamma(shape=a * t**beta - a * (t - 1)**beta, scale = b)
        return x


    mean = 0.4251371301399517                                             # the mean descibes the fianl degradation that the material 
                                                            # will have at the end of the time period  T
    beta = 0.15                                              # beta descibes the curvature and direction of the curve. 
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
        plt.savefig('probability_plot.png', dpi=300)
        # plt.show()

    # Define custom categories
    categories = [10, 20, 100]
    # Categorize deterioration values into custom categories
    Dlabel = deterioration.copy()
    for n in range(N): 
        for t in range(T):
            if deterioration[n,t] <= 0.1: 
                Dlabel[n,t] = 0
            elif  deterioration[n,t] <= 0.2 and deterioration[n,t]> 0.1 :
                Dlabel[n,t] = 1
            elif deterioration[n,t] > 0.2  :
                Dlabel[n,t] = 2
 

    # Calculate transition counts between categories over time
    #Find the number of values in a category in time t 
    #Find the number of values in all the categories in the   
    time_step = step_size

    list1 = []
    for t in range(0, T+1, time_step):
        matrix = np.zeros((3,3))
        list1.append(matrix)
 
    for n in range(N): 
        i = 0  
        for t in range(1,T,time_step): 
            current = int(Dlabel[n,t-1])
            future = int(Dlabel[n,t])
            list1[i][current][future]+=1
            i += 1
    list1 = np.array(list1)


    #lets take a sum of the numbers of each row. 
    #lets divide by the values of the row
    list3 = []
    for t in range(1,len(list1)): 
        new_list = []
        for row in list1[t-1]: 
            sum = np.sum(row)
            if sum!= 0:
                new_row = (row/sum)
            else: 
                new_row = row
            new_list.append(new_row)
        list3.append(new_list)
    
    # Replace NaN and infinite values with 0
    # array = np.array(list1)
    array = np.array(list3)
    array = np.nan_to_num(list3, nan=0, posinf=0, neginf=0) 

    for t in range(len(array)):
        for i in range(len(array[0])):  # Assuming your matrix size is (3, 3)
            s = int(np.sum(array[t][i]))  # Calculate the sum of the row
            if s == 0:
                if i == 0:
                    array[t][i] = np.array([0. , 1. , 0.])
                else: 
                    array[t][i] = np.array([0. , 0. , 1.])

    if save_probabilities :
        np.save('gamma_probabilities.npy', array)

    return array


# if __name__=="__main__":

#     p = matrices_gen(N = 1000000, T= 50, do_plot =True,step_size= 5,SIMPLE_STUFF=True,save_probabilities= True)

#     print(p)



