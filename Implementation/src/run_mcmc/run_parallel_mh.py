import tensorflow_probability as tfp
import numpy as np
import sys
import multiprocessing as mp
import json

sys.path.append("/Users/jay/Desktop/Bachelorarbeit/Implementation")
from src.execute_model import run_model_single_parameter_node
from src.likelihood.likelihood_independent import likelihood_independent
from src.likelihood.likelihood_dependent import likelihood_dependent
from dependencies.mh.mh import MH
from src.construct_model import get_model

runConfigPath = "/Users/jay/Desktop/Bachelorarbeit/run_config.json"
with open(runConfigPath, "r") as file:
    run_config = json.load(file)

configPath = run_config["configPath"]
basis = run_config["basis"]
model = get_model(configPath, basis)


def run_single_chain(
    version,
    init_state,
    iterations,
    param_lower,
    param_upper,
    likelihood_dependence,
    likelihood_sd,
    sd_transition_factor,
    max_probability,
):
    # Define likelihood
    if likelihood_dependence:
        likelihood_function = likelihood_dependent
    else:
        likelihood_function = likelihood_independent

    def likelihood_kernel(param_vec):
        _, y_model, y_observed, _ = run_model_single_parameter_node(model, param_vec)
        return likelihood_function(y_model, y_observed, sd=likelihood_sd)

    def sample_kernel(x):
        return np.random.normal(x, (param_upper - param_lower) / sd_transition_factor)

    parameters_to_sample = tfp.distributions.Uniform(low=param_lower, high=param_upper)
    return MH(
        parameters_to_sample.prob,
        sample_kernel,
        likelihood_kernel,
        init_state,
        iterations,
        param_lower,
        param_upper,
        max_sampling=max_probability,
        version=version,
    )


def run_mcmc_mh_parallel(
    version="ignoring",
    chains=4,
    sd_transition_factor=6,
    likelihood_dependence=False,
    likelihood_sd=1,
    max_probability=False,
    iterations=2500,
    init_method="default",
    custom_init_states=None,
):
    # Construct Params
    configurationObject = model.configurationObject
    param_names = []
    param_lower = []
    param_upper = []
    for param in configurationObject["parameters"]:
        if param["distribution"] == "Uniform":
            param_names.append(param["name"])
            param_lower.append(param["lower"])
            param_upper.append(param["upper"])
        else:
            raise NotImplementedError(
                f"Sorry, the distribution {param['distribution']} is not supported yet"
            )
    param_lower = np.array(param_lower)
    param_upper = np.array(param_upper)

    # Initialize states
    if init_method != "default":
        init_states = custom_init_states
    else:
        init_states = [
            np.random.uniform(low=param_lower, high=param_upper) for _ in range(chains)
        ]

    # Set up pool
    with mp.Pool(chains) as pool:
        results = pool.starmap(
            run_single_chain,
            [
                (
                    version,
                    state,
                    iterations,
                    param_lower,
                    param_upper,
                    likelihood_dependence,
                    likelihood_sd,
                    sd_transition_factor,
                    max_probability,
                )
                for state in init_states
            ],
        )

    return results, iterations * chains
