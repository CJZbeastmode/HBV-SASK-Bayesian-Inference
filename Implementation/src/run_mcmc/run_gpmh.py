import sys

sys.path.append('/Users/jay/Desktop/Bachelorarbeit/Implementation/src')
from execute_model import run_model_single_parameter_node
from likelihood.ll_normmeasured import likelihood_normmeasured
from gpmh_aggregation import GPMH
from construct_model import get_model
import tensorflow_probability as tfp
import numpy as np

model = get_model()

def run_mcmc_gpmh():
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
    n = 2500
    N = 4
    sampled_params = GPMH(target, kernel, likelihood_kernel, init_state, n, N, param_lower, param_upper)
    return sampled_params, n * N
