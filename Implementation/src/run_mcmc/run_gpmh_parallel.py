import numpy as np
from concurrent.futures import ProcessPoolExecutor
import sys

sys.path.append('/Users/jay/Desktop/Bachelorarbeit/Implementation/src')
from execute_model import run_model_single_parameter_node
from likelihood.ll_normmeasured import likelihood_normmeasured
from construct_model import get_model
import tensorflow_probability as tfp
import numpy as np
from multiprocessing import Pool

model = get_model()

configurationObject = model.configurationObject
param_names = []
param_lower = []
param_upper = []
for param in configurationObject["parameters"]:
    # for now the Uniform distribution is only supported
    if param["distribution"] == "Uniform":
        param_names.append(param["name"])
        param_lower.append(param["lower"])
        param_upper.append(param["upper"])
    else:
        raise NotImplementedError(f"Sorry, the distribution {param['distribution']} is not supported yet")
param_lower = np.array(param_lower)
param_upper = np.array(param_upper)

# Define likelihood
likelihood_function = likelihood_normmeasured
def likelihood_kernel(param_vec):
    _, y_model, y_observed, _ = run_model_single_parameter_node(model, param_vec)
    return likelihood_function(y_model, y_observed)

target = tfp.distributions.Uniform(low=param_lower, high=param_upper).prob

def kernel(base):
    return tfp.distributions.Normal(loc=base, scale=(param_upper - param_lower) / 6).sample().numpy()

init_state = [0, 2.5, 2.5, 0.5, 475, 0.5, 0.05]
n = 10
N = 20


def compute_K_values(args):
    Y, N, j = args

    K = np.zeros(N + 1)
    a = Y[j]
    for k in range(N + 1):
        if j == k:
            K[k] = 1
        else:
            b = Y[k]
            K[k] = np.mean(target(b) * likelihood_kernel(b) / (target(a) * likelihood_kernel(a)))
    return K


def GPMH():

    d = len(init_state)
    X = np.zeros((n * (N + 1), d))
    X[0] = init_state
    Y = np.zeros((N + 1, d))
    Y[0] = init_state

    for i in range(n):
        for j in range(1, N + 1):
            samp = kernel(Y[0])

            for ind in range(len(samp)):
                if samp[ind] < param_lower[ind] or samp[ind] > param_upper[ind]:
                    samp = Y[0]
                    break
            Y[j] = samp

        A = np.zeros(N + 1)

        p = Pool()
        results = p.map(compute_K_values, [(Y, N, j) for j in range(N + 1)])
        p.close()
        p.join()

        for j, K in enumerate(results):
            A[j] = np.prod(K)
        
        start_index = i * N
        A = A / np.sum(A)
        sample_indices = np.random.choice(np.arange(0, N + 1), size=N, replace=True, p=A)
        
        for ind in range(N):
            X[start_index + ind] = Y[sample_indices[ind]]

        I = np.random.choice(np.arange(1, N + 1))
        
        Y[0] = Y[I]
        print(f'{i} done')

    return X

def run_mcmc_gpmh():
    sampled_params = GPMH()
    return sampled_params, n * N
