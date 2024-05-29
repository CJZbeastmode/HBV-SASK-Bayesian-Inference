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

#sampling_otb = ['ignoring', 'refl_bound', 'aggr']
#sensitivity_transition = [6, 8, 10, 12, 18, 24]
#sensitivity_likelihood_independent = [1, 3, 5, 8]
sensitivity_likelihood_dependent = [0.2, 0.4]
#max_sampling = [False, True]
#iteration = [5000, 10000, 20000, 40000, 80000]
#burnin_factor = [2, 3, 5]
init_method = ['random', 'min', 'max', 'q1_prior', 'mean_prior', 'q3_prior', 'q1_posterior', 'median_posterior', 'q3_posterior']
to_benchmark = [sensitivity_likelihood_dependent, init_method]
benchmark_data = ['sensitivity_likelihood_dependent', 'init_method']

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
    
    for item in range(len(to_benchmark)):
        test_case = to_benchmark[item]
        test_name = benchmark_data[item]
        results = []

        for case in test_case:
            start = time.time()
            separate_chain = False
            run_mcmc = run_mcmc_mh

            # sd_transition_factor=6, sd_likelihood=1, max_probability=False, iteration=10000
            dictionary = ['ignoring', 6, False, 1, False, 10000, 'random']
            
            if test_name == 'sampling_otb':
                dictionary[0] = case
            if test_name == 'sensitivity_transition':
                dictionary[1] = case
            if test_name == 'sensitivity_likelihood_independent':
                dictionary[2] = False
                dictionary[3] = case
            if test_name == 'sensitivity_likelihood_dependent':
                dictionary[2] = True
                dictionary[3] = case
            if test_name == 'max_sampling':
                dictionary[4] = case
            if test_name == 'iteration':
                dictionary[5] = case
            if test_name == 'init_method':
                dictionary[6] = case

            print(f'Test Case: {test_name}, {case}')
            sampled_params, total_iterations = run_mcmc(version=dictionary[0], \
                                                        sd_transition_factor=dictionary[1], \
                                                        likelihood_dependence=dictionary[2], \
                                                        sd_likelihood=dictionary[3], \
                                                        max_probability=dictionary[4], \
                                                        iteration=dictionary[5], \
                                                        init_method=dictionary[6])
            end = time.time()
            timed = end - start
            print(f'Time needed for test case {case}: {str(timed)}')

            burnin_fac = 5
            if test_name == 'burnin_factor':
                burnin_fac = case
            burnin = int(total_iterations / burnin_fac)

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
            results.append([str(case), rmse_mean, rmse_max, mae_mean, mae_max, timed])

        fmt = '%s,%s,%s,%s,%s,%s'
        np.savetxt(f'{test_name}.txt', results, delimiter=',', fmt=fmt, header='Test_Case, RMSE_Mean, RMSE_Max, MAE_Mean, MAE_Max, Time', comments='')
else:
    pass
