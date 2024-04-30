def simple_model(t, alpha, beta, l):
    return l*np.exp(-alpha*t)*(np.cos(beta*t)+alpha/beta*np.sin(beta*t))

