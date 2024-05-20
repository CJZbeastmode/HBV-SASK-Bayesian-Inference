import sys

sys.path.append('/Users/jay/Desktop/Bachelorarbeit/Implementation/src')
from dependencies.gpmh.gpmh import *
from construct_model import get_model
import numpy as np

configPath = "/Users/jay/Desktop/Bachelorarbeit/Implementation/configurations/config_short.json"
basis = "Oldman_Basin"
model = get_model(configPath, basis)

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


    num_proposals = 100
    num_accepted = 50
    # Simulation configuration
    config = {"NumProposals": num_proposals, "NumAccepted": num_accepted}
    configPath = "/Users/jay/Desktop/Bachelorarbeit/Implementation/configurations/config_short.json"
    basis = "Oldman_Basin"
    problem = AbstractSamplingProblem(configPath, basis)
    gmh_kernel = GMHKernel(config, problem)

    # Initial state
    state = np.random.rand(problem.param_bounds['lower'].size)  # Initialize with a random state within bounds

    # Generate 10,000 samples
    iterations = 200
    samples = []
    for _ in range(iterations):
        accepted_states = gmh_kernel.step(state)
        if accepted_states:  # Only update state if there are accepted states
            state = accepted_states[-1]  # Update state to the last accepted state
        samples.extend(accepted_states)

    print("Number of samples collected:", len(samples))

    return samples, num_accepted * iterations
