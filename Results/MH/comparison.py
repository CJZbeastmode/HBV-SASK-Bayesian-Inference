import pandas as pd
import numpy as np
import sys
import json

# TODO: Change root path
root = "/Users/jay/Desktop/Bachelorarbeit"

primary_output = pd.read_csv(f"{root}/Results/Fundamental/primary_output.out")

tuned_output = pd.read_csv(f"{root}/Results/Fundamental/tuned_output.out")

sys.path.append(f"{root}/Implementation/src")
from execute_model import run_model_single_parameter_node
from construct_model import get_model

testConfigPath = f"{root}/test_config.json"
with open(testConfigPath, "r") as file:
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


def calculate(samples):
    sample_param = []
    for i in range(7):
        sample_param.append(np.random.choice(samples.iloc[:, i], 1000))
    sample_param = np.array(sample_param).T

    posterior = []
    for _, vec in enumerate(sample_param):
        _, y_model, measured_data, _ = run_model_single_parameter_node(
            model, np.array(vec)
        )
        posterior.append(y_model)

    return np.mean(np.array(posterior), axis=0)


_, _, measured_data, _ = run_model_single_parameter_node(model, primary_output.iloc[0])

primary_rmse = []
primary_mae = []
for _ in range(100):
    posterior_mean_primary = calculate(primary_output)
    rmse_val = rmse(posterior_mean_primary, measured_data)
    mae_val = mae(posterior_mean_primary, measured_data)
    primary_rmse.append(rmse_val)
    primary_mae.append(mae_val)

tuned_rmse = []
tuned_mae = []
posterior_mean_tuneds = []
for _ in range(100):
    posterior_mean_tuned = calculate(tuned_output)
    rmse_val = rmse(posterior_mean_tuned, measured_data)
    mae_val = mae(posterior_mean_tuned, measured_data)
    tuned_rmse.append(rmse_val)
    tuned_mae.append(mae_val)


print(f"RMSE of Primary Posterior Mean: {np.array(primary_rmse).mean()}")
print(f"MAE of Primary Posterior Mean: {np.array(primary_mae).mean()}")
print(f"RMSE of Tuned Posterior Mean: {np.array(tuned_rmse).mean()}")
print(f"MAE of Tuned Posterior Mean: {np.array(tuned_mae).mean()}")


"""
RMSE of Primary Posterior Mean: 22.122504129857315
MAE of Primary Posterior Mean: 11.400067417022779
RMSE of Tuned Posterior Mean: 22.124942509212538
MAE of Tuned Posterior Mean: 11.600318945622558
"""
