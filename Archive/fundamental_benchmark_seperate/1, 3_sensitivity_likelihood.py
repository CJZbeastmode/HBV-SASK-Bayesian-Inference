import time
import numpy as np
import pandas as pd
import sys

sys.path.append('/Users/jay/Desktop/Bachelorarbeit/Implementation')
from src.run_mcmc.run_mh import run_mcmc_mh
from src.construct_model import get_model
from src.execute_model import run_model_single_parameter_node

configPath = "/Users/jay/Desktop/Bachelorarbeit/Implementation/configurations/config_train_oldman.json"
basis = "Oldman_Basin"
model = get_model(configPath, basis)

def rmse(result, target):
    diff = result - target
    aggr = 0
    for i in range(len(diff)):
        aggr += diff[i] ** 2
    rmse = (aggr / (len(diff))) ** 0.5
    return rmse

def mae(result, target):
    return np.absolute(result - target).mean()

if __name__ == "__main__": 
    
    test_case = 3

    start = time.time()
    separate_chain = False
    run_mcmc = run_mcmc_mh
    sampled_params, total_iterations = run_mcmc(sd_likelihood=test_case)
    end = time.time()
    timed = end-start
    print(f'Time needed for test case {test_case}: {str(timed)}')

    burnin = int(total_iterations / 5)

    samples = pd.DataFrame(sampled_params, columns=['TT','C0','beta','ETF','FC','FRAC','K2'])
    # Sampling Max
    param_vec = []
    for i in range(len(samples.loc[0])):
        values, counts = np.unique(samples.iloc[:, i], return_counts=True)
        ind = np.argmax(counts)
        param_vec.append(values[ind])
    _, posterior_max, measured_data, _ = run_model_single_parameter_node(model, param_vec)

    # Sampling Mean
    param_vec = []
    for i in range(7):
        param_vec.append(np.random.choice(samples.iloc[:, i], 1000))
    param_vec = np.array(param_vec).T
    posterior = []
    for _, vec in enumerate(param_vec):
        _, y_model, _, _ = run_model_single_parameter_node(model, np.array(vec))
        posterior.append(y_model)
    posterior_mean = np.mean(np.array(posterior), axis=0) 

    rmse_mean = rmse(posterior_mean, measured_data)
    rmse_max = rmse(posterior_max, measured_data)
    mae_mean = mae(posterior_mean, measured_data)
    mae_max = mae(posterior_max, measured_data)
    results = [test_case, rmse_mean, rmse_max, mae_mean, mae_max, timed]

    np.savetxt('3_sensitivity_likelihood.txt', results, delimiter=',', header='Test_Case, RMSE_Mean, RMSE_Max, MAE_Mean, MAE_Max, Time', comments='')

else:
    pass
