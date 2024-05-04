import numpy as np
import math

def GPMH(target, kernel, likelihood_kernel, init_state, n, N, lower_bound, upper_bound):

    d = len(init_state)
    I = 0
    X = np.zeros((n * (N + 1), d))
    
    X[0] = init_state
    Y = np.zeros((N + 1, d))
    Y[0] = init_state

    log_posterior = np.zeros(N + 1)
    A = np.zeros(N + 1)
    K = np.zeros(N + 1)

    for i in range(n):
        for j in range(1, N + 1):
            samp = kernel(X[I])

            for ind in range(len(samp)):
                if samp[ind] < lower_bound[ind] :
                    samp[ind] = lower_bound[ind]
                elif samp[ind] > upper_bound[ind]:
                    samp[ind] = upper_bound[ind]
            Y[j] = samp
    
        # to parallel
        for j in range(N + 1):
            K = np.zeros(N + 1)
            for k in range(N + 1):
                if j == k:
                    K[k] = 1
                else:
                    a = Y[j]
                    b = Y[k]
                    K[k] = np.mean(target(b) * likelihood_kernel(b) / (target(a) * likelihood_kernel(a)))
            A[j] = 1
            for k in range(N + 1):
                A[j] *= K[k]
           
        start_index = i * N
        end_index = (i + 1) * N
        A = A / np.sum(A)
        sample_indices = np.random.choice(np.arange(0, N + 1), size=N, replace=True, p=A)
        
        for ind in range(N):
            X[start_index + ind] = Y[sample_indices[ind]]

        I = np.random.choice(np.arange(start_index, end_index))
        Y[0] = X[I]
        print(f'{i} done')

    return X