import numpy as np
import math
import random

def GPMH(target, kernel, likelihood_kernel, init_state, n, N, lower_bound, upper_bound):

    d = len(init_state)
    I = 0
    X = np.zeros((n * (N + 1), d))
    
    X[0] = init_state
    Y = np.zeros((N + 1, d))
    Y[0] = init_state

    samples = np.zeros((N + 1, d))

    
    for i in range(n):
        # Generate Samples
        samples[0] = X[-1]
        for j in range(1, N + 1):
            samp = kernel(samples[0])

            for ind in range(len(samp)):
                if samp[ind] < lower_bound[ind] or samp[ind] > upper_bound[ind]:
                    samp = samples[0]
                    break
            samples[j] = samp

        # Construct Lookup matrix
        lookup_matrix = np.zeros((N + 1, N + 1))
        # to parallel
        for j in range(N + 1):
            K = np.zeros(N + 1)
            for k in range(N + 1):
                if j == k:
                    K[k] = 1
                else:
                    a = samples[j]
                    b = samples[k]
                    acceptance_rates = np.mean(target(b) * likelihood_kernel(b) / (target(a) * likelihood_kernel(a)))
            lookup_matrix[j] = acceptance_rates
        
        # Decision
        last_index = 0
        for j in range(N):
            this_index = random.randint(0, N - 1)
            
            samp = samples[this_index]
            acceptance_rate = min(1, lookup_matrix[last_index][this_index])
            if np.random.random() >= acceptance_rate:
                Y[j + 1] = Y[j]
            else:
                Y[j + 1] = samples[this_index]
                last_index = this_index
            
        for ind in range(N):
            X[i * N + ind] = Y[ind + 1]

        Y[0] = X[-1]
                
        print(f'{i} done')

    return X



# Decide one by one after generation