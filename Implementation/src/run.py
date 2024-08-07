import time
import numpy as np
import sys
import json

# TODO: Change root path
root = "/Users/jay/Desktop/Bachelorarbeit"

sys.path.append(f"{root}/Implementation")
from src.run_mcmc.run_dream import run_mcmc_dream
from src.run_mcmc.run_gpmh import run_mcmc_gpmh
from src.run_mcmc.run_mh import run_mcmc_mh
from src.run_mcmc.run_parallel_mh import run_mcmc_mh_parallel
from src.construct_model import get_model

runConfigPath = f"{root}/run_config.json"
with open(runConfigPath, "r") as file:
    run_config = json.load(file)

configPath = run_config["configPath"]
basis = run_config["basis"]
model = get_model(configPath, basis)

if __name__ == "__main__":
    mode = run_config["mode"]
    if mode == "mh":
        run_mcmc = run_mcmc_mh
        chain_algo = False
    elif mode == "parallel_mh":
        run_mcmc = run_mcmc_mh_parallel
        chain_algo = True
    elif mode == "gpmh":
        run_mcmc = run_mcmc_gpmh
        chain_algo = False
    elif mode == "dream":
        run_mcmc = run_mcmc_dream
        chain_algo = True
    else:
        print(
            "The algorithm is not implemented. Try one of the following four options:\nmh\nparallel_mh\ngpmh\ndream"
        )
        sys.exit(1)

    start = time.time()

    if "kwargs" in run_config:
        kwargs = run_config["kwargs"]
        sampled_params, total_iterations = run_mcmc(**kwargs)
    else:
        sampled_params, total_iterations = run_mcmc()
    end = time.time()
    print("Time needed: " + str(end - start))

    if chain_algo:
        nchains = len(sampled_params)
    else:
        nchains = 1

    # Post Processing
    if "burnin_fac" in run_config:
        burnin_fac = run_config["burnin_fac"]
    else:
        burnin_fac = 5
    burnin = int(total_iterations / nchains / burnin_fac)
    if chain_algo:
        for i in range(nchains):
            sampled_params[i] = sampled_params[i][burnin:]
    else:
        sampled_params = sampled_params[burnin:]

    if "effective_sample_size" in run_config:
        ess = run_config["effective_sample_size"]
    else:
        ess = 1
    if chain_algo:
        for i in range(nchains):
            sampled_params[i] = sampled_params[i][::ess]
    else:
        sampled_params = sampled_params[::ess]

    # Save Data
    output_file_name = (
        (run_config["output_file_name"] + ".out")
        if "output_file_name" in run_config
        else "mcmc_data.out"
    )
    if chain_algo:
        if "separate_chains" in run_config:
            separate_chains = run_config["separate_chains"]
        else:
            separate_chains = False

        if separate_chains:
            samples = np.hstack(sampled_params)
            header = ""
            for i in range(nchains):
                if i != 0:
                    header = header + ","
                header = (
                    header
                    + f"TT_{i + 1},C0_{i + 1},beta_{i + 1},ETF_{i + 1},FC_{i + 1},FRAC_{i + 1},K2_{i + 1}"
                )
            np.savetxt(
                output_file_name, samples, delimiter=",", header=header, comments=""
            )
        else:
            samples = np.concatenate(sampled_params)
            header = "TT,C0,beta,ETF,FC,FRAC,K2"
            np.savetxt(
                output_file_name, samples, delimiter=",", header=header, comments=""
            )
    else:
        header = "TT,C0,beta,ETF,FC,FRAC,K2"
        np.savetxt(
            output_file_name, sampled_params, delimiter=",", header=header, comments=""
        )
