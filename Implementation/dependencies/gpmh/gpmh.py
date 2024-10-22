import numpy as np
import tensorflow_probability as tfp

tfd = tfp.distributions

import sys

# TODO: Change root path
root = "/Users/jay/Desktop/Bachelorarbeit"

sys.path.append(f"{root}/Implementation/src")
from execute_model import run_model_single_parameter_node
from likelihood.likelihood_independent import likelihood_independent
from likelihood.likelihood_dependent import likelihood_dependent


class SamplingState:
    def __init__(self, state, meta=None):
        self.state = np.array(state)
        self.meta = meta if meta is not None else {}

    def has_meta(self, key):
        return key in self.meta


# Constructing abstract sampling problem
class AbstractSamplingProblem:
    def __init__(self, model, likelihood_dependence, likelihood_sd):
        self.model = model
        self.param_bounds = self.get_param_bounds()
        self.uniform_distribution = tfd.Uniform(
            low=self.param_bounds["lower"], high=self.param_bounds["upper"]
        )
        self.likelihood_dependence = likelihood_dependence
        self.likelihood_sd = likelihood_sd

    def get_param_bounds(self):
        configurationObject = self.model.configurationObject
        param_lower = []
        param_upper = []
        for param in configurationObject["parameters"]:
            if param["distribution"] == "Uniform":
                param_lower.append(param["lower"])
                param_upper.append(param["upper"])
            else:
                raise NotImplementedError(
                    f"Sorry, the distribution {param['distribution']} is not supported yet"
                )
        return {"lower": np.array(param_lower), "upper": np.array(param_upper)}

    # Calculate log density
    def log_density(self, state):
        if state is None:
            return -np.inf
        _, y_model, y_observed, _ = run_model_single_parameter_node(
            self.model, state.state
        )
        if self.likelihood_dependence:
            likelihood_function = likelihood_dependent
        else:
            likelihood_function = likelihood_independent
        return likelihood_function(y_model, y_observed, sd=self.likelihood_sd)


# Kernel of GPMH
class GMHKernel:
    def __init__(self, num_proposals, num_accepted, problem, sampling_kernel):
        self.problem = problem
        self.sampling_kernel = sampling_kernel
        self.num_proposals = num_proposals
        self.num_accepted = num_accepted
        self.proposed_states = []

    # Propose multiple states
    def serial_proposal(self, state):
        if not state.has_meta("LogTarget"):
            state.meta["LogTarget"] = self.problem.log_density(state)

        self.proposed_states = [state]
        for _ in range(1, self.num_proposals + 1):
            new_state = self.sampling_kernel(state)
            if new_state:
                new_state.meta["LogTarget"] = self.problem.log_density(new_state)
            self.proposed_states.append(new_state)

        r = np.array(
            [s.meta["LogTarget"] if s else -np.inf for s in self.proposed_states]
        )
        self.acceptance_density(r)

    # Calculate acceptance density
    def acceptance_density(self, r):
        a = np.zeros((self.num_proposals + 1, self.num_proposals + 1))
        for i in range(self.num_proposals + 1):
            for j in range(self.num_proposals + 1):
                if i != j and self.proposed_states[j] is not None:
                    a[i, j] = np.exp(r[j] - r[i])
                    a[i, j] = min(1.0, a[i, j] / (self.num_proposals + 1))
                a[i, i] -= a[i, j]
        self.stationary_acceptance = self.compute_stationary_acceptance(a)

    # Compute stationary acceptance rate
    def compute_stationary_acceptance(self, a):
        mat = np.zeros((self.num_proposals + 2, self.num_proposals + 1))
        mat[: self.num_proposals + 1, :] = a.T - np.eye(self.num_proposals + 1)
        mat[-1, :] = 1
        rhs = np.zeros(self.num_proposals + 2)
        rhs[-1] = 1
        return np.linalg.lstsq(mat, rhs, rcond=None)[0]

    # Sample for one step
    def step(self, state):
        self.serial_proposal(SamplingState(state))
        return self.sample_stationary()

    def sample_stationary(self):
        probability = self.stationary_acceptance / np.sum(self.stationary_acceptance)
        indices = np.random.choice(
            self.num_proposals + 1, self.num_accepted, p=probability
        )
        return [
            self.proposed_states[i].state
            for i in indices
            if self.proposed_states[i] is not None
        ]
