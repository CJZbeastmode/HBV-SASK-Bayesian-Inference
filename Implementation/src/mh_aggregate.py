import numpy as np

def MH(target, kernel, likelihood_kernel, init_state, n, lower_bound, upper_bound):
    samples = [np.array(init_state)]
    num_accept = 0
    for _ in range(n):
        #sample candidate from normal distribution
        a = samples[-1]
        b = np.random.normal(a, [8/6, 5/6, 3/6, 1/6, 950/6, 0.8/6, 0.1/6])
        
        #calculate probability of accepting this candidate
        for i in range(len(b)):
            if b[i] < lower_bound[i]:
                b[i] = lower_bound[i]
            elif b[i] > upper_bound[i]:
                b[i] = upper_bound[i]
        
        p = target(b) * likelihood_kernel(b) / (target(a) * likelihood_kernel(a))
        p = p.numpy()
        prob = min(1, np.mean(p)) #max, min
        if np.random.random() >= prob:
            samples.append(samples[-1])         
        else:
            samples.append(b)
            num_accept += 1
            
    return samples
