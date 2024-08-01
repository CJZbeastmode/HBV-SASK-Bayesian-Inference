from math import pi, exp


def likelihood_gaussian(y_model, y_observed):
    err_var = 5  # sigma**2
    n = len(y_observed)
    prod = 1
    for i in range(n):
        t1 = 1 / ((2 * pi * err_var) ** 0.5)
        t2 = -((y_model[i] - y_observed[i]) ** 2) / (2 * err_var)
        t2 = exp(t2)
        term = t1 * t2
        prod = prod * term
    return prod
    