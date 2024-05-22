import numpy as np
import tensorflow_probability as tfp

def likelihood_normmeasured(y_model, y_observed):
    like_ctot = tfp.distributions.Normal(loc=y_observed, scale=1)
    logp = np.sum(like_ctot.log_prob(y_model))
    print("logp: " + str(logp))
    if np.isnan(logp):
        logp = -np.inf
    return logp

# independant
