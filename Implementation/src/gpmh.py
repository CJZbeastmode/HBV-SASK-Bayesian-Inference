import numpy as np
import math

def GPMH(target, kernel, likelihood_kernel, init_state, n, N = 8):

    d = len(init_state)
    I = 0
    X = np.zeros((n * (N + 1), d))
    
    X[0] = init_state
    Y = np.zeros((N + 1, d))
    Y[0] = init_state

    for i in range(1, N + 1):
        Y[i] = kernel(X[0])

    log_posterior = np.zeros(N + 1)
    A = np.zeros(N + 1)
    K = np.zeros(N + 1)

    for i in range(n):
        for j in range(N + 1):
            #log_post_prob = likelihood_kernel(Y[j]) + target(Y[j])
            #log_posterior[j] = 0 if log_post_prob == float('-inf') or log_post_prob == float('inf') else log_post_prob
            for k in range(N + 1):
                K[k] = likelihood_kernel(Y[j], Y[k])
            tg = target(Y[j])
            prod_tg = 1
            for l in range(len(tg)):
                prod_tg *= tg[l]
            A[j] = np.prod(K[:j]) * np.prod(K[j + 1:]) * prod_tg
            #A[j] = 0 if A[j] == float('-inf') or A[j] == float('inf') else A[j]

           
        start_index = (i - 1) * N + i
        end_index = i * (N + 1) + 1
        A = A / np.sum(A)
        sample_indices = np.random.choice(np.arange(0, N + 1), size=(end_index - start_index), replace=True, p=A)
        for ind in range(end_index - start_index):
            X[start_index + ind] = Y[sample_indices[ind]]

        I = np.random.choice(np.arange(start_index, end_index))
        Y[0] = X[I]
        for j in np.arange(1, N):
            Y[j] = kernel(X[I])
        print(f'{i} done')

    return X