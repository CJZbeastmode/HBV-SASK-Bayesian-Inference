import numpy as np
import tensorflow_probability as tfp


def likelihood_independent(y_model, y_observed, sd=1):
    norm = tfp.distributions.Normal(loc=y_observed, scale=sd)
    logp = np.sum(norm.log_prob(y_model))
    if np.isnan(logp):
        logp = -np.inf
    return logp


# independant
# dependent scale: Vrugt
