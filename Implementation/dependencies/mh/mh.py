import numpy as np
import math

def MH(proposal, sample_kernel, likelihood_kernel, init_state, n, lower_bounds, upper_bounds):
    samples = [np.array(init_state)]
    num_accept = 0
    for _ in range(n):
        #sample candidate from normal distribution
        a = samples[-1]
        b = sample_kernel(a)
        p = [0, 0, 0, 0, 0, 0, 0]
        
        #calculate probability of accepting this candidate

        invalid = False
        reject = False

        for i in range(len(b)):
            if b[i] < lower_bounds[i] or b[i] > upper_bounds[i]:
                invalid = True
                break
        
        if invalid:
            reject = True
        else:
            p = np.log(proposal(b)) - np.log(proposal(a)) + likelihood_kernel(b) - likelihood_kernel(a)
            prob = min(1, math.e ** np.mean(p)) #max, min
            if np.random.random() >= prob:
                reject = True
            
        if not reject:
            samples.append(b)
            num_accept += 1
        else:
            samples.append(samples[-1])

    return samples
