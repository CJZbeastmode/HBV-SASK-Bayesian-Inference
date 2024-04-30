
from likelihood.ll_loggaussian import likelihood_loggaussian

likelihood_function = likelihood_loggaussian
GR = False
separate_chain = True
save_image = True
mode = 'DREAM'


def likelihood_kernel(param_vec):
    _, y_model, y_observed, _ = run_model_single_parameter_node(model, param_vec)
    return likelihood_function(y_model, y_observed)

def mcmc():
    parameters_to_sample = SampledParam(tfp.distributions.Uniform, low=param_lower, high=param_upper)
    sampled_parameter = [parameters_to_sample]
    niterations = 10000
    converged = False
    total_iterations = niterations
    nchains = 5

    sampled_params, _ = run_dream(sampled_parameter, likelihood_kernel, niterations=niterations, \
                                           nchains=nchains, multitry=False, gamma_levels=4, adapt_gamma=True, \
                                            history_thin=1, model_name='test_mcmc_hydrological', verbose=True)
    return sampled_params, total_iterations

"""
def mcmc():
    init_state = [0, 2.5, 2.5, 0.5, 475, 0.5, 0.05]
    n = 10000
    N = 4
    sampled_params = GPMH(target, kernel, likelihood_kernel, init_state, n, N)
"""

