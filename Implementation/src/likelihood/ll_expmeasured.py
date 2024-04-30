import numpy as np
from scipy.stats import expon
from math import log

def likelihood_expmeasured(y_model, y_observed):
    exponent = log(y_observed[-1] - y_observed[0])
    like_ctot = expon(scale=exponent)
    logp = np.sum(like_ctot.logpdf(y_model))
    print("logp: " + str(logp))
    if np.isnan(logp):
        logp = -np.inf
    return logp
