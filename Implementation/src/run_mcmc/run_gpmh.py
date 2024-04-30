import sys

sys.path.append('/Users/jay/Desktop/Bachelorarbeit/Implementation/src')
from execute_model import run_model_single_parameter_node
from likelihood.ll_normmeasured import likelihood_normmeasured
from gpmh import GPMH
from src.construct_model import get_model

model = get_model()

def run_mcmc_gpmh():
    # Define likelihood
    likelihood_function = likelihood_normmeasured
    def likelihood_kernel(param_vec):
        _, y_model, y_observed, _ = run_model_single_parameter_node(model, param_vec)
        return likelihood_function(y_model, y_observed)


    init_state = [0, 2.5, 2.5, 0.5, 475, 0.5, 0.05]
    n = 10000
    N = 4
    sampled_params = GPMH(target, kernel, likelihood_kernel, init_state, n, N)
    return sampled_params, n
