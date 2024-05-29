import tensorflow_probability as tfp
import numpy as np
import pandas as pd
import sys

sys.path.append('/Users/jay/Desktop/Bachelorarbeit/Implementation')
from src.execute_model import run_model_single_parameter_node
from src.likelihood.likelihood_independent import likelihood_independent
from src.likelihood.likelihood_dependent import likelihood_dependent
from dependencies.mh.mh import MH
from dependencies.mh.mh_refl_boundary import MH_RB
from dependencies.mh.mh_aggregate import MH_A
from src.construct_model import get_model

configPath = "/Users/jay/Desktop/Bachelorarbeit/Implementation/configurations/config_train_oldman.json"
basis = "Oldman_Basin"
posterior_rudimentary = pd.read_csv('/Users/jay/Desktop/Bachelorarbeit/Implementation/src/run_mcmc/posterior_rudimentary.csv').apply(pd.to_numeric, errors='coerce')

model = get_model(configPath, basis)

def run_mcmc_mh(version='ignoring', sd_transition_factor=6, likelihood_dependence=False, sd_likelihood=1, max_probability=False, iteration=10000, init_method='random'):
    
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
    if likelihood_dependence:
        likelihood_function = likelihood_dependent
    else:
        likelihood_function = likelihood_independent
    def likelihood_kernel(param_vec):
        _, y_model, y_observed, _ = run_model_single_parameter_node(model, param_vec)
        return likelihood_function(y_model, y_observed, sd=sd_likelihood)
    
    def sample_kernel(x):
        return np.random.normal(x, (param_upper - param_lower) / sd_transition_factor)

    parameters_to_sample = tfp.distributions.Uniform(low=param_lower, high=param_upper)

    if init_method == 'random':
        init_state = [0, 0, 0, 0, 0, 0, 0]
        for _ in range(1000):
            init_state += np.random.uniform(low=param_lower, high=param_upper)
        init_state /= 1000
    elif init_method == 'min':
        init_state = param_lower
    elif init_method == 'max':
        init_state = param_upper
    elif init_method == 'q1_prior':
        init_state = param_lower + ((param_upper - param_lower) / 4)
    elif init_method == 'mean_prior':
        init_state = (param_upper - param_lower) / 2
    elif init_method == 'q3_prior':
        init_state = param_lower + ((param_upper - param_lower) * 3 / 4)
    elif init_method == 'q1_posterior':
        init_state = np.array(posterior_rudimentary.iloc[1].values[1:])
    elif init_method == 'median_posterior':
        init_state = np.array(posterior_rudimentary.iloc[2].values[1:])
    elif init_method == 'q3_posterior':
        init_state = np.array(posterior_rudimentary.iloc[3].values[1:])


    if version == 'ignoring':
        x = MH(parameters_to_sample.prob, sample_kernel, likelihood_kernel, init_state, iteration, param_lower, param_upper, max_sampling=max_probability)
    elif version == 'refl_bound':
        x = MH_RB(parameters_to_sample.prob, sample_kernel, likelihood_kernel, init_state, iteration, param_lower, param_upper, max_sampling=max_probability)
    elif version == 'aggr':
        x = MH_A(parameters_to_sample.prob, sample_kernel, likelihood_kernel, init_state, iteration, param_lower, param_upper, max_sampling=max_probability)
    else:
        print('This variant has not been implemented')

    return x, iteration
