import numpy as np

def MH(target, sample_kernel, likelihood_kernel, init_state, n, lower_bounds, upper_bounds):
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
            p = target(b) * likelihood_kernel(b) / (target(a) * likelihood_kernel(a))
            p = p.numpy()
            prob = min(1, np.mean(p)) #max, min
            if np.random.random() >= prob:
                reject = True
            
        if not reject:
            samples.append(b)
            num_accept += 1
        else:
            samples.append(samples[-1])

    return samples
