import numpy as np


def MH(
    proposal,
    sample_kernel,
    likelihood_kernel,
    init_state,
    n,
    lower_bounds,
    upper_bounds,
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

        invalid = False
        reject = False

        for i in range(len(b)):
            if b[i] < lower_bounds[i] or b[i] > upper_bounds[i]:
                invalid = True
                break

        if invalid:
            reject = True
        else:
            # log_p = np.log(proposal(b)) - np.log(proposal(a)) + likelihood_kernel(b) - likelihood_kernel(a)
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
                reject = True

        if not reject:
            samples.append(b)
            num_accept += 1
        else:
            samples.append(samples[-1])

    return samples
