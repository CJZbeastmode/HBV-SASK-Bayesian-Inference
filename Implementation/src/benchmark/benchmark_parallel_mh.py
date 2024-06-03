import time
import numpy as np
import pandas as pd
import sys

sys.path.append('/Users/jay/Desktop/Bachelorarbeit/Implementation')
from src.run_mcmc.run_parallel_mh import run_mcmc_mh_parallel
from src.construct_model import get_model
from src.execute_model import run_model_single_parameter_node
from dependencies.PyDREAM.pydream.convergence import Gelman_Rubin

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

    test_cases = [[1, 10000], [2, 5000], [4, 2500], [5, 2000], [8, 1250], [10, 1000], [20, 500], [50, 200]]
    res = []

    for case in test_cases:
        num_chains = case[0]
        iterations = case[1]
        start = time.time()
        results = run_mcmc_mh_parallel(num_chains=num_chains, iterations=iterations, max_probability=True, sd_transition_factor=8, likelihood_dependence=True, sd_likelihood=0.6)
        end = time.time()
        timed = end - start
        print("Time needed: " + str(timed))

        # Convergence
        gr = Gelman_Rubin(np.array(results))
        gr_mean = gr.mean()
        gr_min = gr.min()
        gr_max = gr.max()

        MAEs_mean = []
        MAEs_max = []
        RMSEs_mean = []
        RMSEs_max = []
        # Mean Accuracy of each case
        for i in range(num_chains):
            samples = pd.DataFrame(results[i])

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

            RMSEs_mean.append(rmse(posterior_mean, measured_data))
            RMSEs_max.append(rmse(posterior_max, measured_data))
            MAEs_mean.append(mae(posterior_mean, measured_data))
            MAEs_max.append(mae(posterior_max, measured_data))

        mae_mean_sep = np.array(MAEs_mean).mean()
        mae_max_sep = np.array(MAEs_max).mean()
        rmse_mean_sep = np.array(RMSEs_mean).mean()
        rmse_max_sep = np.array(RMSEs_max).mean()



        # Combine
        burnin = int(iterations / 2)
        combined_results = []
        results = np.array(results)
        for i in range(num_chains):
            combined_results.append(results[i][burnin:, :])
        combined_results = np.concatenate(combined_results)

        # Sampling Max
        samples = pd.DataFrame(combined_results)
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

        rmse_mean_combined = rmse(posterior_mean, measured_data)
        rmse_max_combined = rmse(posterior_max, measured_data)
        mae_mean_combined = mae(posterior_mean, measured_data)
        mae_max_combined = mae(posterior_max, measured_data)

        res.append([num_chains, gr_min, gr_mean, gr_max, rmse_mean_sep, rmse_max_sep, mae_mean_sep, mae_max_sep, rmse_mean_combined, rmse_max_combined, mae_mean_combined, mae_max_combined, timed])

        # Backup
        fmt = '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s'
        np.savetxt(f'{num_chains}_chains.txt', res, delimiter=',', fmt=fmt, header='Num Chains, GR_Min, GR_Mean, GR_Max, RMSE_Mean_Sep, RMSE_Max_Sep, MAE_Mean_Sep, MAE_Max_Sep, RMSE_Mean_Combined, RMSE_Max_Combined, MAE_Mean_Combined, MAE_Max_Combined, Time', comments='')

    fmt = '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s'
    np.savetxt('benchmark_parallel_mh.txt', res, delimiter=',', fmt=fmt, header='Num Chains, GR_Min, GR_Mean, GR_Max, RMSE_Mean_Sep, RMSE_Max_Sep, MAE_Mean_Sep, MAE_Max_Sep, RMSE_Mean_Combined, RMSE_Max_Combined, MAE_Mean_Combined, MAE_Max_Combined, Time', comments='')

else:
    pass
