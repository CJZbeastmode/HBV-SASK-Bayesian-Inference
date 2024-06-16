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


"""
def calculate_gaussian_likelihood(y_t_observed, y_t_model, error_variance=5.0):
    if y_t_observed is not None and y_t_model is not None:
        exponent = -0.5 * ((y_t_observed - y_t_model) ** 2) / error_variance
        likelihood = np.exp(exponent) / np.sqrt(2 * np.pi * error_variance)
        return likelihood  # in case when y_t_observed and y_t_model are lists the likelihood will be a list as well
    else:
        return 0
"""
