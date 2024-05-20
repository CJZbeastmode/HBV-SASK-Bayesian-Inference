import tensorflow_probability as tfp
import numpy as np
import sys

sys.path.append('/Users/jay/Desktop/Bachelorarbeit/Implementation/src')
from execute_model import run_model_single_parameter_node
from dependencies.PyDREAM.pydream.parameters import SampledParam
from likelihood.ll_normmeasured import likelihood_normmeasured
from dependencies.PyDREAM.pydream.core import run_dream
from src.construct_model import get_model

configPath = "/Users/jay/Desktop/Bachelorarbeit/Implementation/configurations/config_short.json"
basis = "Oldman_Basin"

model = get_model(configPath, basis)

# Define Likelihood
def likelihood_kernel(param_vec):
    likelihood_function = likelihood_normmeasured
    _, y_model, y_observed, _ = run_model_single_parameter_node(model, param_vec)
    return likelihood_function(y_model, y_observed)


def run_mcmc_dream():
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

    # Run
    niterations = 10000
    nchains = 5
    parameters_to_sample = SampledParam(tfp.distributions.Uniform, low=param_lower, high=param_upper)
    #The run_dream function expects a list rather than a single variable
    sampled_parameter = [parameters_to_sample]
    sampled_params, log_ps = run_dream(sampled_parameter, likelihood_kernel, niterations=niterations, nchains=nchains, multitry=False, gamma_levels=4, adapt_gamma=True, history_thin=1, model_name='test_mcmc_hydrological', verbose=True)
    return sampled_params, niterations

