import numpy as np

def MH2(target, kernel, likelihood_kernel, init_state, n):
    samples = [np.array(init_state)]
    num_accept = 0
    for _ in range(n):
        #sample candidate from normal distribution
        a = samples[-1]
        b = np.random.normal(a, [0.3, 0.2, 0.1, 0.04, 40, 0.03, 0.004])
        
        #calculate probability of accepting this candidate
        p = target(b) * likelihood_kernel(b) / (target(a) * likelihood_kernel(a))
        p = p.numpy()
        prob = min(1, np.mean(p))
        if np.random.random() < prob:
            samples.append(b)
            num_accept += 1
        else:
            samples.append(a)

    return samples
