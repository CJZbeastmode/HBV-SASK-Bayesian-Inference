from math import log, pi

def likelihood_loggaussian(y_model, y_observed):
    err_var = 5 # sigma**2
    n = len(y_observed)
    t1 = - (n / 2.0) * log(2 * pi)

    t2 = (n / 2) * log(err_var)

    t3 = 0
    for i in range(n):
        t3 += (y_model[i] - y_observed[i]) ** 2
    t3 *= 1 / (2 * err_var)
    return t1 - t2 - t3
    
