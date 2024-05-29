import numpy as np
import tensorflow_probability as tfp
tfd = tfp.distributions

import sys
sys.path.append('/Users/jay/Desktop/Bachelorarbeit/Implementation/src')
from execute_model import run_model_single_parameter_node
from likelihood.likelihood_independent import likelihood_independent
from construct_model import get_model

class SamplingState:
    def __init__(self, state, meta=None):
        self.state = np.array(state)
        self.meta = meta if meta is not None else {}

    def has_meta(self, key):
        return key in self.meta

class MCMCProposal:
    def __init__(self, bounds):
        self.bounds = bounds
        self.scale = (bounds['upper'] - bounds['lower']) / 6  # Scale for proposal distribution

    def sample(self, state):
        new_state = np.random.normal(loc=state.state, scale=self.scale)
        # Reflective boundary handling
        for i in range(len(new_state)):
            if new_state[i] < self.bounds['lower'][i]:
                new_state[i] = self.bounds['lower'][i] + (self.bounds['lower'][i] - new_state[i])
            elif new_state[i] > self.bounds['upper'][i]:
                new_state[i] = self.bounds['upper'][i] - (new_state[i] - self.bounds['upper'][i])
        return SamplingState(new_state)

class AbstractSamplingProblem:
    def __init__(self, configPath, basis):
        self.model = get_model(configPath, basis)
        self.param_bounds = self.get_param_bounds()
        self.uniform_distribution = tfd.Uniform(low=self.param_bounds['lower'], high=self.param_bounds['upper'])

    def get_param_bounds(self):
        configurationObject = self.model.configurationObject
        param_lower = []
        param_upper = []
        for param in configurationObject["parameters"]:
            if param["distribution"] == "Uniform":
                param_lower.append(param["lower"])
                param_upper.append(param["upper"])
            else:
                raise NotImplementedError(f"Sorry, the distribution {param['distribution']} is not supported yet")
        return {'lower': np.array(param_lower), 'upper': np.array(param_upper)}

    def log_density(self, state):
        if state is None:
            return -np.inf
        _, y_model, y_observed, _ = run_model_single_parameter_node(self.model, state.state)
        likelihood_function = likelihood_independent
        return likelihood_function(y_model, y_observed)

class GMHKernel:
    def __init__(self, config, problem, proposal=None):
        self.problem = problem
        self.proposal = proposal if proposal else MCMCProposal(problem.param_bounds)
        self.num_proposals = config.get("NumProposals", 1)
        self.num_accepted = config.get("NumAccepted", self.num_proposals)
        self.proposed_states = []

    def serial_proposal(self, state):
        if not state.has_meta("LogTarget"):
            state.meta["LogTarget"] = self.problem.log_density(state)

        self.proposed_states = [state]
        for _ in range(1, self.num_proposals + 1):
            new_state = self.proposal.sample(state)
            if new_state:
                new_state.meta["LogTarget"] = self.problem.log_density(new_state)
            self.proposed_states.append(new_state)

        r = np.array([s.meta["LogTarget"] if s else -np.inf for s in self.proposed_states])
        self.acceptance_density(r)

    def acceptance_density(self, r):
        a = np.zeros((self.num_proposals + 1, self.num_proposals + 1))
        for i in range(self.num_proposals + 1):
            for j in range(self.num_proposals + 1):
                if i != j and self.proposed_states[j] is not None:
                    a[i, j] = np.exp(r[j] - r[i])
                    a[i, j] = min(1.0, a[i, j] / (self.num_proposals + 1))
                a[i, i] -= a[i, j]
        self.stationary_acceptance = self.compute_stationary_acceptance(a)

    def compute_stationary_acceptance(self, a):
        mat = np.zeros((self.num_proposals + 2, self.num_proposals + 1))
        mat[:self.num_proposals + 1, :] = a.T - np.eye(self.num_proposals + 1)
        mat[-1, :] = 1
        rhs = np.zeros(self.num_proposals + 2)
        rhs[-1] = 1
        return np.linalg.lstsq(mat, rhs, rcond=None)[0]

    def step(self, state):
        self.serial_proposal(SamplingState(state))
        return self.sample_stationary()

    def sample_stationary(self):
        probability = self.stationary_acceptance / np.sum(self.stationary_acceptance)
        indices = np.random.choice(self.num_proposals + 1, self.num_accepted, p=probability)
        return [self.proposed_states[i].state for i in indices if self.proposed_states[i] is not None]
