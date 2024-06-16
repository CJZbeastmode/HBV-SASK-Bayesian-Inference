import numpy as np
import math


def MH_A(
    proposal,
    sample_kernel,
    likelihood_kernel,
    init_state,
    n,
    lower_bound,
    upper_bound,
    max_sampling=False,
):
    samples = [np.array(init_state)]
    num_accept = 0
    for _ in range(n):
        # sample candidate from normal distribution
        a = samples[-1]
        b = sample_kernel(a)
        p = [0, 0, 0, 0, 0, 0, 0]

        # calculate probability of accepting this candidate
        for i in range(len(b)):
            if b[i] < lower_bound[i]:
                b[i] = lower_bound[i]
            elif b[i] > upper_bound[i]:
                b[i] = upper_bound[i]

        p = (
            proposal(b)
            * np.exp(likelihood_kernel(b))
            / (proposal(a) * np.exp(likelihood_kernel(a)))
        )
        if max_sampling:
            prob = min(1, np.max(p))
        else:
            prob = min(1, np.mean(p))
        if np.random.random() >= prob:
            samples.append(samples[-1])
        else:
            samples.append(b)
            num_accept += 1

    return samples
