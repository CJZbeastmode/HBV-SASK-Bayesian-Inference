import sys
import pandas as pd
import json

# TODO: Change root path
root = "/Users/jay/Desktop/Bachelorarbeit"

sys.path.append(f"{root}/Implementation/src")
from dependencies.gpmh.gpmh import *
from src.construct_model import get_model
import numpy as np


def run_mcmc_gpmh(
    num_proposals=8,
    num_accepted=4,
    likelihood_dependence=False,
    likelihood_sd=1,
    sd_transition_factor=6,
    version="ignoring",
    init_method="random",
    iterations=2500,
):
    posterior_rudimentary = pd.read_csv(
        f"{root}/Implementation/src/run_mcmc/posterior_rudimentary.csv"
    ).apply(pd.to_numeric, errors="coerce")

    runConfigPath = f"{root}/run_config.json"
    with open(runConfigPath, "r") as file:
        run_config = json.load(file)

    configPath = run_config["configPath"]
    basis = run_config["basis"]
    model = get_model(configPath, basis)

    problem = AbstractSamplingProblem(model, likelihood_dependence, likelihood_sd)

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

    def sampling_kernel(
        state,
        param_lower=param_lower,
        param_upper=param_upper,
        sd=sd_transition_factor,
        version=version,
    ):
        new_state = np.random.normal(
            loc=state.state, scale=(param_upper - param_lower) / sd
        )
        if version == "ignoring":
            for i in range(len(new_state)):
                if new_state[i] < param_lower[i] or new_state[i] > param_upper[i]:
                    new_state = state.state
                    break
        elif version == "refl_bound":
            for i in range(len(new_state)):
                if new_state[i] < param_lower[i]:
                    new_state[i] = param_lower[i] + (param_lower[i] - new_state[i])
                elif new_state[i] > param_upper[i]:
                    new_state[i] = param_upper[i] - (new_state[i] - param_upper[i])
        elif version == "aggr":
            for i in range(len(new_state)):
                if new_state[i] < param_lower[i]:
                    new_state[i] = param_lower[i]
                elif new_state[i] > param_upper[i]:
                    new_state[i] = param_upper[i]
        return SamplingState(new_state)

    gmh_kernel = GMHKernel(num_proposals, num_accepted, problem, sampling_kernel)

    # Initial state
    if init_method == "random":
        state = [0, 0, 0, 0, 0, 0, 0]
        for _ in range(1000):
            state += np.random.uniform(low=param_lower, high=param_upper)
        state /= 1000
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

    samples = []
    for iter in range(iterations):
        accepted_states = gmh_kernel.step(state)
        # Update states
        if accepted_states:
            state = accepted_states[-1]
        samples.extend(accepted_states)
        print(f"{iter} done")

    print("Number of samples collected:", len(samples))

    return samples, num_accepted * iterations
