import time
import numpy as np
import pandas as pd
import sys

sys.path.append('/Users/jay/Desktop/Bachelorarbeit/Implementation')
from src.run_mcmc.run_dream import run_mcmc_dream
from src.construct_model import get_model
from src.execute_model import run_model_single_parameter_node
from dependencies.PyDREAM.pydream.convergence import Gelman_Rubin

configPath = "/Users/jay/Desktop/Bachelorarbeit/Implementation/configurations/config_train_oldman.json"
basis = "Oldman_Basin"
model = get_model(configPath, basis)
monte_carlo_size = 1000

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

    num_chains = 4
    iterations = 2500

    start = time.time()
    results, _ = run_mcmc_dream(niterations=iterations, nchains=num_chains)
    results = np.array(results)
    end = time.time()
    timed = end - start
    print("Time needed: " + str(timed))

    # Convergence
    gr = np.array(Gelman_Rubin(np.array(results)))
    gr_reps = 1

    while np.any(gr > 1.2):
        toResampleIndex = []
        for i in range(7):
            if gr[i] > 1.2:
                toResampleIndex.append(i)
        start = time.time()
        temp_res = run_mcmc_dream(niterations=int(iterations/5), nchains=num_chains)
        end = time.time()
        timed = timed + (end - start)

        results = np.hstack([results, np.array(temp_res)])
        results = results[:, int(iterations/5):]
        gr = np.array(Gelman_Rubin(np.array(results)))
        gr_reps += 1

    gr = np.vstack([np.array(['TT','C0','beta','ETF','FC','FRAC','K2']), gr]).T
    fmt = '%s,%s'
    np.savetxt(f'{num_chains}_chains_gr.txt', gr, fmt=fmt, delimiter=',', header='Parameter,GR', comments='')

    for i in range(num_chains):
        np.savetxt(f'{num_chains}_chains_result_{i + 1}.txt', results[i], delimiter=',', header='TT,C0,beta,ETF,FC,FRAC,K2', comments='')

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
            param_vec.append(np.random.choice(samples.iloc[:, i], monte_carlo_size))
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

    res = np.vstack([np.array(range(num_chains)) + 1, np.array(RMSEs_mean).T, np.array(RMSEs_max).T, np.array(MAEs_mean).T, np.array(MAEs_max).T]).T

    np.savetxt(f'{num_chains}_sep_accuracy.txt', res, delimiter=',', header='Chain,RMSE_Mean_Sep,RMSE_Max_Sep,MAE_Mean_Sep,MAE_Max_Sep', comments='')

    # Combine
    burnin = int(iterations / 5)
    ess = 3
    combined_results = []
    results = np.array(results)
    for i in range(num_chains):
        burned_in = results[i][burnin:, :]
        burned_in = burned_in[::ess]
        combined_results.append(burned_in)
    combined_results = np.concatenate(combined_results)
    np.savetxt(f'{num_chains}_chains_result_combined.txt', combined_results, delimiter=',', header='TT,C0,beta,ETF,FC,FRAC,K2', comments='')


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
        param_vec.append(np.random.choice(samples.iloc[:, i], monte_carlo_size))
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

    res = np.vstack([num_chains, rmse_mean_combined, rmse_max_combined, mae_mean_combined, mae_max_combined, timed, gr_reps]).T
    np.savetxt(f'{num_chains}_chains_analysis.txt', res, delimiter=',', header='Num_Chains,RMSE_Mean_Combined,RMSE_Max_Combined,MAE_Mean_Combined,MAE_Max_Combined,Time,GR_Reps', comments='')

else:
    pass
