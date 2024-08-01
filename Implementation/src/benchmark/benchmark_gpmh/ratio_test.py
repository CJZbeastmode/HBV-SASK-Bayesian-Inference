import time
import numpy as np
import pandas as pd
import sys
import json

# TODO: Change root path
root = "/Users/jay/Desktop/Bachelorarbeit"

sys.path.append(f"{root}/Implementation")
from src.run_mcmc.run_gpmh import run_mcmc_gpmh
from src.construct_model import get_model
from src.execute_model import run_model_single_parameter_node

runConfigPath = f"{root}/run_config.json"
with open(runConfigPath, "r") as file:
    run_config = json.load(file)

configPath = run_config["configPath"]
basis = run_config["basis"]
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
    # 1, Ratio Test
    test_cases = [[5, 5], [10, 5], [20, 5], [40, 5], [80, 5]]
    iterations = 2000
    res = []

    for case in test_cases:
        num_proposals = case[0]
        num_accepted = case[1]
        start = time.time()
        results, _ = run_mcmc_gpmh(
            num_proposals=num_proposals,
            num_accepted=num_accepted,
            likelihood_dependence=False,
            likelihood_sd=1,
            sd_transition_factor=6,
            version="ignoring",
            init_method="random",
            iterations=iterations,
        )
        end = time.time()
        timed = end - start
        print("Time needed: " + str(timed))

        burnin = int(iterations * num_accepted / 2)
        results = np.array(results)[burnin:, :]

        # Sampling Max
        samples = pd.DataFrame(results)
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
        ratio = num_proposals / num_accepted

        res.append(
            [
                num_proposals / num_accepted,
                rmse_mean,
                rmse_max,
                mae_mean,
                mae_max,
                timed,
            ]
        )

        # Backup
        fmt = "%s,%s,%s,%s,%s,%s"
        np.savetxt(
            f"ratio_{ratio}.txt",
            res,
            delimiter=",",
            fmt=fmt,
            header="Ratio, RMSE_Mean, RMSE_Max, MAE_Mean, MAE_Max, Time",
            comments="",
        )

    fmt = "%s,%s,%s,%s,%s,%s"
    np.savetxt(
        "ratio_test_parallel_gpmh.txt",
        res,
        delimiter=",",
        fmt=fmt,
        header="Ratio, RMSE_Mean, RMSE_Max, MAE_Mean, MAE_Max, Time",
        comments="",
    )

else:
    pass
