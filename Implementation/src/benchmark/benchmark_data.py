import time
import json
import numpy as np
import pandas as pd
import sys

sys.path.append("/Users/jay/Desktop/Bachelorarbeit/Implementation")
from src.run_mcmc.run_dream import run_mcmc_dream
from src.construct_model import get_model
from src.execute_model import run_model_single_parameter_node

runConfigPath = "/Users/jay/Desktop/Bachelorarbeit/test_config.json"
with open(runConfigPath, "r") as file:
    run_config = json.load(file)

configPath = run_config["configPath"]
basis = run_config["basis"]
model = get_model(configPath, basis)


"""
Testing (on Oldman)
1, time series containing floods — (1994-1996)
2, on short term (regular) — (2004)

Testing (on Banff): only for case 4 ()

Training
1, using short term not on floods, using short term on floods: 1983-85. O vs. 2004-06. O
2, using long term not on floods, using long term on floods:  1982-90. O vs. 1998-06. O
3, short term vs long term (containing floods): 2004-06. O vs 1998-06. O
4, Oldman vs. Banff: 1997-03 O vs. 1997-03 B
5, tuning phase: 25%, 50%, and 100%: 2000-08 O
"""

make_file_struct_oldman = lambda filename : [f'/Users/jay/Desktop/Bachelorarbeit/Implementation/configurations/benchmark_configs/{filename}.json', 'Oldman_Basin', filename]
make_file_struct_banff = lambda filename : [f'/Users/jay/Desktop/Bachelorarbeit/Implementation/configurations/benchmark_configs/{filename}.json', 'Banff_Basin', filename]

short_no_floods = make_file_struct_oldman('short_no_floods')
short_floods = make_file_struct_oldman('short_floods')
long_no_floods = make_file_struct_oldman('long_no_floods')
long_floods = make_file_struct_oldman('long_floods')
oldman_train = make_file_struct_oldman('oldman_train')
banff_train = make_file_struct_banff('banff_train')
su25 = make_file_struct_oldman('su25')
su50 = make_file_struct_oldman('su50')
su100 = make_file_struct_oldman('su100')

test_oldman_long = make_file_struct_oldman('test_oldman_long')
test_oldman_short = make_file_struct_oldman('test_oldman_short')
test_banff = make_file_struct_banff('test_banff')
testing_data_regular = [test_oldman_long, test_oldman_short]
testing_data_regular_name = ['test_oldman_long', 'test_oldman_short']
testing_data_special = [test_oldman_long, test_banff]
testing_data_special_name = ['test_oldman_long', 'test_banff']

short_train = [short_no_floods, short_floods]
long_train = [long_no_floods, long_floods]
length_train = [short_floods, long_floods]
data_train = [oldman_train, banff_train]
spin_up_train = [su25, su50, su100]
to_benchmark = [short_train, long_train, length_train, data_train, spin_up_train]
benchmark_data = ['short_train', 'long_train', 'length_train', 'data_train', 'spin_up_train']


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
            model = get_model(case[0], case[1])

            chain_iterations = 1250 # TODO
            nchains = 8
            total_iterations = chain_iterations * nchains

            start = time.time()
            run_mcmc = run_mcmc_dream
            sampled_params, _ = run_mcmc(chains=nchains, iterations=chain_iterations)
            print(np.array(sampled_params).shape)

            end = time.time()
            timed = end - start

            burnin_fac = 5
            burnin = int(chain_iterations / burnin_fac)
            for i in range(nchains):
                sampled_params[i] = np.array(sampled_params[i])[burnin:]

            ess = 3
            for i in range(nchains):
                sampled_params[i] = sampled_params[i][::ess]
            print(np.array(sampled_params).shape)

            sampled_params = np.vstack(sampled_params)
            print(sampled_params.shape)

            samples = pd.DataFrame(
                sampled_params, columns=["TT", "C0", "beta", "ETF", "FC", "FRAC", "K2"]
            )

            # Sampling Meanprint(sampled_params.shape)
            param_vec = []
            for i in range(7):
                param_vec.append(np.random.choice(samples.iloc[:, i], 1000)) # TODO
            param_vec = np.array(param_vec).T

            testing_data = testing_data_special if test_name == 'data_train' else testing_data_regular
            testing_data_name = testing_data_special_name if test_name == 'data_train' else testing_data_regular_name

            for scenario in range(len(testing_data)):
                test_model = get_model(testing_data[scenario][0], testing_data[scenario][1])
                posterior = []
                for _, vec in enumerate(param_vec):
                    _, y_model, measured_data, _ = run_model_single_parameter_node(test_model, np.array(vec))
                    posterior.append(y_model)
                posterior_mean = np.mean(np.array(posterior), axis=0)
                rmse_mean = rmse(posterior_mean, measured_data)
                mae_mean = mae(posterior_mean, measured_data)
                results.append(
                    [test_name, case[2], testing_data_name[scenario], rmse_mean, mae_mean, timed]
                )

                # Backup
                fmt = "%s,%s,%s,%s,%s,%s"
                np.savetxt(
                    "backup.txt",
                    results,
                    delimiter=",",
                    fmt=fmt,
                    header="Test_Name,Train_Data,Test_Data,MAE_Mean,Time",
                    comments="",
                )

            # Backup
            fmt = "%s,%s,%s,%s,%s,%s"
            np.savetxt(
                "backup.txt",
                results,
                delimiter=",",
                fmt=fmt,
                    header="Test_Name,Train_Data,Test_Data,MAE_Mean,Time",
                comments="",
            )

        # Backup
            fmt = "%s,%s,%s,%s,%s,%s"
            np.savetxt(
                "backup.txt",
                results,
                delimiter=",",
                fmt=fmt,
                    header="Test_Name,Train_Data,Test_Data,MAE_Mean,Time",
                comments="",
            )

    # Overall
    fmt = "%s,%s,%s,%s,%s,%s"
    np.savetxt(
        f"benchmark_data.txt",
        results,
        delimiter=",",
        fmt=fmt,
        header="Test_Name,Train_Data,Test_Data,MAE_Mean,Time",
        comments="",
    )

else:
    pass
