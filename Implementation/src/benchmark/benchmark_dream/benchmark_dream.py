import time
import numpy as np
import pandas as pd
import sys
import json

# TODO: Change root path
root = "/Users/jay/Desktop/Bachelorarbeit"

sys.path.append(f"{root}/Implementation")
from src.run_mcmc.run_dream import run_mcmc_dream
from src.construct_model import get_model
from src.execute_model import run_model_single_parameter_node

runConfigPath = f"{root}/run_config.json"
with open(runConfigPath, "r") as file:
    run_config = json.load(file)

configPath = run_config["configPath"]
basis = run_config["basis"]
model = get_model(configPath, basis)

niterations = 1250 
nchains = 8 
monte_carlo_number = 1000

DEpairs = [1, 2, 3]
multitry = [False, 3, 5, 7, 10]
hardboundaries = [False, True]
crossover_burnin = [None, 0.1, 0.2, 0.5]
adapt_crossover = [False, True]
nCR = [1, 3, 5]
snooker = [0, 0.2, 0.5, 0.8, 1]
p_gamma_unity = [0, 0.2, 0.5, 0.8, 1]
sensitivity_likelihood_independent = [1, 3, 5, 8]
sensitivity_likelihood_dependent = [0.2, 0.4, 0.6, 0.8]
burnin_factor = [2, 3, 5]
effective_sample_size = [1, 2, 3, 4, 5]
init_method = [
    "random",
    "min",
    "max",
    "q1_prior",
    "mean_prior",
    "q3_prior",
    "q1_posterior",
    "median_posterior",
    "q3_posterior",
]
to_benchmark = [DEpairs, multitry, hardboundaries, crossover_burnin, adapt_crossover, nCR, snooker, p_gamma_unity, sensitivity_likelihood_independent, sensitivity_likelihood_dependent, burnin_factor, effective_sample_size, init_method]
benchmark_data = ['DEpairs', 'multitry', 'hardboundaries', 'crossover_burnin', 'adapt_crossover', 'nCR', 'snooker', 'p_gamma_unity', 'sensitivity_likelihood_independent', 'sensitivity_likelihood_dependent', 'burnin_factor', 'effective_sample_size', 'init_method']


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
    results = []

    for item in range(len(to_benchmark)):
        test_case = to_benchmark[item]
        test_name = benchmark_data[item]

        for case in test_case:
            start = time.time()
            separate_chain = False
            run_mcmc = run_mcmc_dream

            dictionary = [1, False, True, None, True, 3, 0.1, 0.2, False, 1, "random"]

            if test_name == "DEpairs":
                dictionary[0] = case
                if case == 2:
                    nchains = 5
                    niterations = 2000
                if case == 3:
                    nchains = 8
                    niterations = 1250
            if test_name == "multitry":
                dictionary[1] = case
            if test_name == "hardboundaries":
                dictionary[2] = case
            if test_name == "crossover_burnin":
                dictionary[3] = case
            if test_name == "adapt_crossover":
                dictionary[4] = case
            if test_name == "nCR":
                dictionary[5] = case
            if test_name == "snooker":
                dictionary[6] = case
            if test_name == "p_gamma_unity":
                dictionary[7] = case
            if test_name == "sensitivity_likelihood_independent":
                dictionary[8] = False
                dictionary[9] = case
            if test_name == "sensitivity_likelihood_dependent":
                dictionary[8] = True
                dictionary[9] = case
            if test_name == "init_method":
                dictionary[10] = case

            sampled_params, _ = run_mcmc(
                niterations=niterations,
                nchains=nchains,
                DEpairs=dictionary[0],
                multitry=dictionary[1],
                hardboundaries=dictionary[2],
                crossover_burnin=dictionary[3],
                adapt_crossover=dictionary[4],
                nCR=dictionary[5],
                snooker=dictionary[6],
                p_gamma_unity=dictionary[7],
                likelihood_dependence=dictionary[8],
                likelihood_sd=dictionary[9],
                init_method=dictionary[10],
            )
            total_iterations = niterations * nchains
            end = time.time()
            timed = end - start
            print(sampled_params)
            print(f"Time needed for test case {case}: {str(timed)}")

            # Combine
            burnin_fac = 5
            if test_name == "burnin_factor":
                burnin_fac = case
            burnin = int(niterations / burnin_fac)
            ess = 1
            if test_name == "effective_sample_size":
                ess = case

            combined_results = []
            for i in range(nchains):
                burned_in = sampled_params[i][burnin:, :]
                burned_in = burned_in[::ess]
                combined_results.append(burned_in)
            combined_results = np.concatenate(combined_results)
            samples = pd.DataFrame(
                combined_results,
                columns=["TT", "C0", "beta", "ETF", "FC", "FRAC", "K2"],
            )
            samples.to_csv("temp.csv")

            # Sampling Max
            param_vec = []
            for i in range(len(samples.loc[0])):
                values, counts = np.unique(samples.iloc[:, i], return_counts=True)
                ind = np.argmax(counts)
                param_vec.append(values[ind])
            _, posterior_max, measured_data, _ = run_model_single_parameter_node(
                model, param_vec
            )

            # Sampling Mean
            param_vec = []
            for i in range(7):
                param_vec.append(
                    np.random.choice(samples.iloc[:, i], monte_carlo_number)
                )
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
            results.append(
                [test_name, case, rmse_mean, rmse_max, mae_mean, mae_max, timed]
            )

            # Backup
            fmt = "%s,%s,%s,%s,%s,%s,%s"
            np.savetxt(
                f"{test_name}.txt",
                results,
                delimiter=",",
                fmt=fmt,
                header="Test_Name, Test_Case, RMSE_Mean, RMSE_Max, MAE_Mean, MAE_Max, Time",
                comments="",
            )

        # Backup
        fmt = "%s,%s,%s,%s,%s,%s,%s"
        np.savetxt(
            f"{test_name}.txt",
            results,
            delimiter=",",
            fmt=fmt,
            header="Test_Name, Test_Case, RMSE_Mean, RMSE_Max, MAE_Mean, MAE_Max, Time",
            comments="",
        )

    # Overall
    fmt = "%s,%s,%s,%s,%s,%s,%s"
    np.savetxt(
        f"benchmark_mh.txt",
        results,
        delimiter=",",
        fmt=fmt,
        header="Test_Name, Test_Case, RMSE_Mean, RMSE_Max, MAE_Mean, MAE_Max, Time",
        comments="",
    )

else:
    pass
