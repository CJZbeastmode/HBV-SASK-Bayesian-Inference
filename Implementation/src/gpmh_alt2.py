import numpy as np
import math

def GPMH(target, kernel, likelihood_kernel, init_state, n, N, lower_bound, upper_bound):

    d = len(init_state)
    I = 0
    X = np.zeros((n * (N + 1), d))
    
    X[0] = init_state
    Y = np.zeros((N, d))
    Y[0] = init_state
    samples = np.zeros((N, d))

    for i in range(n):
        for j in range(N):
            samp = kernel(X[I])

            for ind in range(len(samp)):
                if samp[ind] < lower_bound[ind] or samp[ind] > upper_bound[ind]:
                    samp = X[I]
                    break
            samples[j] = samp
    
        lookup_matrix = np.zeros(N)
        for j in range(N):
            lookup_matrix[j] = np.mean(target(samples[j]) * likelihood_kernel(samples[j]) / (target(X[-1]) * likelihood_kernel(X[-1])))


        # Acceptance
        last_sample = X[-1]
        for j in range(N):
            acceptance_rate = min(lookup_matrix[j], 1)
            if np.random.random() >= acceptance_rate:
                Y[j] = last_sample
            else:
                Y[j] = samples[j]

        for j in range(N):
            X[N * i + j] = Y[j]

        print(f'{i} done')

    return X