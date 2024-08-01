import tensorflow_probability as tfp
import numpy as np
import pandas as pd
import sys
import json

# TODO: Change root path
root = "/Users/jay/Desktop/Bachelorarbeit"

sys.path.append(f"{root}/Implementation/")
from src.execute_model import run_model_single_parameter_node
from dependencies.PyDREAM.pydream.parameters import SampledParam
from src.likelihood.likelihood_independent import likelihood_independent
from src.likelihood.likelihood_dependent import likelihood_dependent
from dependencies.PyDREAM.pydream.core import run_dream
from src.construct_model import get_model

posterior_rudimentary = pd.read_csv(
    f"{root}/Implementation/src/run_mcmc/posterior_rudimentary.csv"
).apply(pd.to_numeric, errors="coerce")

runConfigPath = f"{root}/run_config.json"
with open(runConfigPath, "r") as file:
    run_config = json.load(file)

configPath = run_config["configPath"]
basis = run_config["basis"]
model = get_model(configPath, basis)


# Define Likelihood
def default_likelihood_kernel(param_vec):
    likelihood_function = likelihood_independent
    _, y_model, y_observed, _ = run_model_single_parameter_node(model, param_vec)
    return likelihood_function(y_model, y_observed)


def likelihood_d2(param_vec):
    likelihood_function = lambda y_model, y_observed: likelihood_dependent(
        y_model, y_observed, sd=0.2
    )
    _, y_model, y_observed, _ = run_model_single_parameter_node(model, param_vec)
    return likelihood_function(y_model, y_observed)


def likelihood_d4(param_vec):
    likelihood_function = lambda y_model, y_observed: likelihood_dependent(
        y_model, y_observed, sd=0.4
    )
    _, y_model, y_observed, _ = run_model_single_parameter_node(model, param_vec)
    return likelihood_function(y_model, y_observed)


def likelihood_d6(param_vec):
    likelihood_function = lambda y_model, y_observed: likelihood_dependent(
        y_model, y_observed, sd=0.6
    )
    _, y_model, y_observed, _ = run_model_single_parameter_node(model, param_vec)
    return likelihood_function(y_model, y_observed)


def likelihood_d8(param_vec):
    likelihood_function = lambda y_model, y_observed: likelihood_dependent(
        y_model, y_observed, sd=0.8
    )
    _, y_model, y_observed, _ = run_model_single_parameter_node(model, param_vec)
    return likelihood_function(y_model, y_observed)


def likelihood_i1(param_vec):
    likelihood_function = lambda y_model, y_observed: likelihood_independent(
        y_model, y_observed, sd=1
    )
    _, y_model, y_observed, _ = run_model_single_parameter_node(model, param_vec)
    return likelihood_function(y_model, y_observed)


def likelihood_i3(param_vec):
    likelihood_function = lambda y_model, y_observed: likelihood_independent(
        y_model, y_observed, sd=3
    )
    _, y_model, y_observed, _ = run_model_single_parameter_node(model, param_vec)
    return likelihood_function(y_model, y_observed)


def likelihood_i5(param_vec):
    likelihood_function = lambda y_model, y_observed: likelihood_independent(
        y_model, y_observed, sd=5
    )
    _, y_model, y_observed, _ = run_model_single_parameter_node(model, param_vec)
    return likelihood_function(y_model, y_observed)


def likelihood_i8(param_vec):
    likelihood_function = lambda y_model, y_observed: likelihood_independent(
        y_model, y_observed, sd=8
    )
    _, y_model, y_observed, _ = run_model_single_parameter_node(model, param_vec)
    return likelihood_function(y_model, y_observed)


def run_mcmc_dream(iterations=1250, chains=8, **kwargs):
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

    # Parameter Assignment
    DEpairs = kwargs["DEpairs"] if "DEpairs" in kwargs else 1
    multitry = kwargs["multitry"] if "multitry" in kwargs else False
    hardboundaries = kwargs["hardboundaries"] if "hardboundaries" in kwargs else True
    crossover_burnin = kwargs["crossover_burnin"] if "crossover_burnin" in kwargs else 0
    nCR = kwargs["nCR"] if "nCR" in kwargs else 3
    snooker = kwargs["snooker"] if "snooker" in kwargs else 0
    p_gamma_unity = kwargs["p_gamma_unity"] if "p_gamma_unity" in kwargs else 0
    init_method = kwargs["init_method"] if "init_method" in kwargs else "not specified"

    l_k = default_likelihood_kernel
    if "likelihood_sd" in kwargs:
        if "likelihood_dependence" not in kwargs:
            print("Failure")
            sys.exit

        if kwargs["likelihood_dependence"]:
            if kwargs["likelihood_sd"] == 0.2:
                l_k = likelihood_d2
            elif kwargs["likelihood_sd"] == 0.4:
                l_k = likelihood_d4
            elif kwargs["likelihood_sd"] == 0.6:
                l_k = likelihood_d6
            elif kwargs["likelihood_sd"] == 0.8:
                l_k = likelihood_d8
        else:
            if kwargs["likelihood_sd"] == 1:
                l_k = likelihood_i1
            elif kwargs["likelihood_sd"] == 3:
                l_k = likelihood_i3
            elif kwargs["likelihood_sd"] == 5:
                l_k = likelihood_i5
            elif kwargs["likelihood_sd"] == 8:
                l_k = likelihood_i8

    randomStart = False
    
    # Initial state
    if init_method == "random" or init_method == "not specified":
        state = param_lower
        randomStart = True
    elif init_method == "min":
        state = param_lower
    elif init_method == "max":
        state = param_upper
    elif init_method == "q1_prior":
        state = param_lower + ((param_upper - param_lower) / 4)
    elif init_method == "mean_prior":
        state = (param_upper - param_lower) / 2
    elif init_method == "q3_prior":
        state = param_lower + ((param_upper - param_lower) * 3 / 4)
    elif init_method == "q1_posterior":
        state = np.array(posterior_rudimentary.iloc[1].values[1:])
    elif init_method == "median_posterior":
        state = np.array(posterior_rudimentary.iloc[2].values[1:])
    elif init_method == "q3_posterior":
        state = np.array(posterior_rudimentary.iloc[3].values[1:])

    states = []
    for _ in range(chains):
        states.append(state)

    # Run
    parameters_to_sample = SampledParam(
        tfp.distributions.Uniform, low=param_lower, high=param_upper
    )
    sampled_parameter = [parameters_to_sample]

    sampled_params, _ = run_dream(
        sampled_parameter,
        l_k,
        niterations=iterations,
        nchains=chains,
        start=states,
        DEpairs=DEpairs,
        multitry=multitry,
        hardboundaries=hardboundaries,
        crossover_burnin=crossover_burnin,
        nCR=nCR,
        snooker=snooker,
        p_gamma_unity=p_gamma_unity,
        randomStart=randomStart,
    )

    return sampled_params, chains * iterations
