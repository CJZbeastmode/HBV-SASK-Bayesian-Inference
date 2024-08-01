import time
import json
import numpy as np
import pandas as pd
import sys

# TODO: Change root path
root = "/Users/jay/Desktop/Bachelorarbeit"

sys.path.append(f"{root}/Implementation")
from src.run_mcmc.run_dream import run_mcmc_dream
from src.construct_model import get_model
from src.execute_model import run_model_single_parameter_node

runConfigPath = f"{root}/test_config.json"
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

make_file_struct_oldman = lambda filename: [
    f"{root}/Implementation/configurations/benchmark_configs/{filename}.json",
    "Oldman_Basin",
    filename,
]
make_file_struct_banff = lambda filename: [
    f"{root}/Implementation/configurations/benchmark_configs/{filename}.json",
    "Banff_Basin",
    filename,
]

short_no_floods = make_file_struct_oldman("short_no_floods")
short_floods = make_file_struct_oldman("short_floods")
long_no_floods = make_file_struct_oldman("long_no_floods")
long_floods = make_file_struct_oldman("long_floods")
oldman_train = make_file_struct_oldman("oldman_train")
banff_train = make_file_struct_banff("banff_train")
su25 = make_file_struct_oldman("su25")
su50 = make_file_struct_oldman("su50")
su100 = make_file_struct_oldman("su100")
all_files = [
    short_no_floods,
    short_floods,
    long_no_floods,
    long_floods,
    oldman_train,
    banff_train,
    su25,
    su50,
    su100,
]


test_oldman_long = make_file_struct_oldman("test_oldman_long")
test_oldman_short = make_file_struct_oldman("test_oldman_short")
test_banff = make_file_struct_banff("test_banff")
testing_data_regular = [test_oldman_long, test_oldman_short]
testing_data_regular_name = ["test_oldman_long", "test_oldman_short"]
testing_data_special = [test_oldman_long, test_banff]
testing_data_special_name = ["test_oldman_long", "test_banff"]


if __name__ == "__main__":
    results = []
    for item in all_files:

        file_path = item[0]
        basin = item[1]
        file_name = item[2]

        model = get_model(file_path, basin)

        chain_iterations = 1250
        nchains = 8
        total_iterations = chain_iterations * nchains

        start = time.time()
        run_mcmc = run_mcmc_dream
        sampled_params, _ = run_mcmc(chains=nchains, iterations=chain_iterations)

        end = time.time()
        timed = end - start

        burnin_fac = 5
        burnin = int(chain_iterations / burnin_fac)
        for i in range(nchains):
            sampled_params[i] = np.array(sampled_params[i])[burnin:]

        ess = 3
        for i in range(nchains):
            sampled_params[i] = sampled_params[i][::ess]

        sampled_params = np.vstack(sampled_params)

        # Overall
        np.savetxt(
            f"{file_name}.txt",
            sampled_params,
            delimiter=",",
            header="TT,C0,beta,ETF,FC,FRAC,K2",
            comments="",
        )

else:
    pass
