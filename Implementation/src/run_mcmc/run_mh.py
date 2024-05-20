import tensorflow_probability as tfp
import numpy as np
import sys

sys.path.append('/Users/jay/Desktop/Bachelorarbeit/Implementation')
from src.execute_model import run_model_single_parameter_node
from src.likelihood.ll_normmeasured import likelihood_normmeasured
from dependencies.mh.mh import MH
from src.construct_model import get_model

configPath = "/Users/jay/Desktop/Bachelorarbeit/Implementation/configurations/config_short.json"
basis = "Oldman_Basin"

model = get_model(configPath, basis)

def run_mcmc_mh():
    # Construct Params
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
    
    def sample_kernel(x):
        return np.random.normal(x, (param_upper - param_lower) / 6)

    parameters_to_sample = tfp.distributions.Uniform(low=param_lower, high=param_upper)
    init_state = np.random.uniform(low=param_lower, high=param_upper)
    x = MH(parameters_to_sample.prob, sample_kernel, likelihood_kernel, init_state, 10000, param_lower, param_upper)
    return x, 10000
