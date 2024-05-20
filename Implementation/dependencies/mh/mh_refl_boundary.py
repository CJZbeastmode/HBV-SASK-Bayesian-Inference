import numpy as np
import math

def MH(proposal, sample_kernel, likelihood_kernel, init_state, n, lower_bound, upper_bound):
    samples = [np.array(init_state)]
    num_accept = 0
    for _ in range(n):
        #sample candidate from normal distribution
        a = samples[-1]
        b = sample_kernel(a)
        p = [0, 0, 0, 0, 0, 0, 0]
        
        #calculate probability of accepting this candidate
        for i in range(len(b)):
            if b[i] < lower_bound[i] or b[i] > upper_bound[i]:
                temp = 2 * a[i] - b[i]
                b[i] = temp
    
        p = np.log(proposal(b)) - np.log(proposal(a)) + likelihood_kernel(b) - likelihood_kernel(a)
        prob = min(1, math.e ** np.mean(p)) #max, min
        if np.random.random() >= prob:
            samples.append(samples[-1])         
        else:
            samples.append(b)
            num_accept += 1
            
    return samples
