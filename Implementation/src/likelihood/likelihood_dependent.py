import numpy as np
import tensorflow_probability as tfp


def likelihood_dependent(y_model, y_observed, sd=0.2):
    norm = tfp.distributions.Normal(loc=y_observed, scale=sd * y_observed)
    logp = np.sum(norm.log_prob(y_model))
    if np.isnan(logp):
        logp = -np.inf
    return logp
