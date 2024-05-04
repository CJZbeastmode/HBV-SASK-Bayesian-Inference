import numpy as np
from scipy.stats import norm

def likelihood_normmeasured(y_model, y_observed):
    like_ctot = norm(loc=y_observed)
    logp = np.sum(like_ctot.logpdf(y_model))
    print("logp: " + str(logp))
    if np.isnan(logp):
        logp = -np.inf
    return logp

# independant
