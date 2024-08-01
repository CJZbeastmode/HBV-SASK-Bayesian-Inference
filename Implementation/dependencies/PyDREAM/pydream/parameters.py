# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf


class SampledParam:
    """A SciPy-based parameter prior class.

    Parameters
    ----------
    scipy_distribution: SciPy continuous random variable class
        A SciPy statistical distribution (i.e. scipy.stats.norm)
    args:
        Arguments for the SciPy distribution
    kwargs:
        keyword arguments for the SciPy distribution

    """

    def __init__(self, tfp_distribution, *args, **kwargs):
        self.dist = tfp_distribution(*args, **kwargs)
        self.dsize = self.random().size

    def interval(self, alpha=1):
        """Return the interval for a given alpha value."""

        lower_quantile = (1 - alpha) / 2
        upper_quantile = 1 - lower_quantile
        lower = self.dist.quantile(int(lower_quantile))
        upper = self.dist.quantile(int(upper_quantile))
        res = np.stack([lower, upper], axis=0)
        for i in range(len(res)):
            res[i] = np.array(res[i])
        return res

    def random(self, reseed=False):
        """Return a random value drawn from this prior."""
        if reseed:
            random_seed = np.random.RandomState()
            tf.random.set_seed(random_seed.random())
        else:
            random_seed = None

        return self.dist.sample().numpy()

    def prior(self, q0):
        """Return the prior log probability given a point.

        Parameters
        ----------
        q0: array
            A location in parameter space.
        """
        logp = np.sum(self.dist.log_prob(q0))

        return logp


class FlatParam(SampledParam):
    """A Flat parameter class (returns 0 at all locations).

    Parameters
    ----------
    test_value: array
        Representative value for the parameter.  Used to infer the parameter dimension, which is needed in the DREAM algorithm.

    """

    def __init__(self, test_value):
        self.dsize = test_value.size

    def prior(self, q0):
        return 0

    def interval(self, alpha=1):
        """Return the interval for a given alpha value."""

        lower = [-np.inf] * self.dsize
        upper = [np.inf] * self.dsize
        return [lower, upper]
