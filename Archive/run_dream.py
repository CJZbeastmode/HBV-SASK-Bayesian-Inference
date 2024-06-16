import pathlib
import time
import pandas as pd
import numpy as np
from scipy.stats import norm, uniform
import seaborn as sns
from matplotlib import pyplot as plt
from math import log, pi, exp
import tensorflow_probability as tfp

import sys

sys.path.append("/Users/jay/Desktop/Bachelorarbeit/Implementation/dependencies")
from hbv_sask.model import HBVSASKModel as hbvmodel
from PyDREAM.pydream.core import run_dream
from PyDREAM.pydream.parameters import SampledParam
from PyDREAM.pydream.convergence import Gelman_Rubin

# TODO - change these paths accordingly
# sys.path.insert(1, '/work/ga45met/Hydro_Models/HBV-SASK-py-tool')
# sys.path.insert(1, '/work/ga45met/mnt/linux_cluster_2/UQEF-Dynamic')


TIME_COLUMN_NAME = "TimeStamp"
INDEX_COLUMN_NAME = "Index_run"
PLOT_FORCING_DATA = True
PLOT_ALL_THE_RUNS = True
QOI_COLUMN_NAME = "model"  # "Value"
MODEL = "hbv-sask"  #'hbv-sask' # 'banchmark_model' or 'simple_model' or 'hbv-sask'
QOI_COLUMN_NAME = "Q_cms"
QOI_COLUM_NAME_MESURED = "streamflow"

# TODO: Measured Data
measured_data = np.array(
    [
        9.50,
        9.18,
        8.85,
        7.78,
        7.01,
        7.53,
        7.31,
        6.76,
        6.60,
        7.07,
        9.94,
        10.70,
        9.58,
        8.53,
        8.86,
        8.73,
        10.10,
        9.72,
        10.30,
        10.50,
        10.30,
        9.99,
        9.65,
        10.10,
        10.30,
        12.50,
        15.10,
        16.30,
        17.00,
        22.80,
        27.00,
        26.20,
    ]
)


def calculate_gaussian_likelihood(y_t_observed, y_t_model, error_variance=5.0):
    """
    Computing Gaussian like likelihood, in case one has measured/observed data
    """
    if y_t_observed is not None and y_t_model is not None:
        exponent = -0.5 * ((y_t_observed - y_t_model) ** 2) / error_variance
        likelihood = np.exp(exponent) / np.sqrt(2 * np.pi * error_variance)
        return likelihood  # in case when y_t_observed and y_t_model are lists the likelihood will be a list as well
    else:
        return 0


# def simple_model(t, alpha, beta, l):
#     return l*np.exp(-alpha*t)*(np.cos(beta*t)+alpha/beta*np.sin(beta*t))


def run_model_single_parameter_node(
    model,
    parameter_value_dict,
    unique_index_model_run=0,
    qoi_column_name=QOI_COLUMN_NAME,
    qoi_column_name_measured=QOI_COLUM_NAME_MESURED,
    **kwargs,
):
    # take_direct_value should be True if parameter_value_dict is a dict with keys being paramter name and values being parameter values;
    # if parameter_value_dict is a list of parameter values corresponding to the order of the parameters in the configuration file, then take_direct_value should be False
    # it is assumed that model is a subclass of HydroModel from UQEF-Dynamic
    results_list = model.run(
        i_s=[
            unique_index_model_run,
        ],
        parameters=[
            parameter_value_dict,
        ],
        createNewFolder=False,
        take_direct_value=True,
        merge_output_with_measured_data=True,
    )
    # extract y_t produced by the model
    y_t_model = results_list[0][0]["result_time_series"][qoi_column_name].to_numpy()
    if (
        qoi_column_name_measured is not None
        and qoi_column_name_measured in results_list[0][0]["result_time_series"]
    ):
        y_t_observed = results_list[0][0]["result_time_series"][
            qoi_column_name_measured
        ].to_numpy()
        # y_t_observed = model.time_series_measured_data_df[qoi_column_name_measured].values
    else:
        y_t_observed = None
    return unique_index_model_run, y_t_model, y_t_observed, parameter_value_dict


# Defining paths and Creating Model Object
# TODO - change these paths accordingly
hbv_model_data_path = pathlib.Path(
    "/Users/jay/Desktop/Bachelorarbeit/Implementation/dependencies/hbv_sask/data"
)
configurationObject = pathlib.Path(
    "/Users/jay/Desktop/Bachelorarbeit/Implementation/configurations/configuration_hbv_6D.json"
)
inputModelDir = hbv_model_data_path
basis = "Oldman_Basin"  # 'Banff_Basin'
workingDir = (
    hbv_model_data_path / basis / "model_runs" / "running_the_model_parallel_simple"
)

# creating HBVSASK model object
writing_results_to_a_file = False
plotting = False
createNewFolder = False  # create a separate folder to save results for each model run
model = hbvmodel.HBVSASKModel(
    configurationObject=configurationObject,
    inputModelDir=inputModelDir,
    workingDir=workingDir,
    basis=basis,
    writing_results_to_a_file=writing_results_to_a_file,
    plotting=plotting,
)


directory_for_saving_plots = workingDir
if not str(directory_for_saving_plots).endswith("/"):
    directory_for_saving_plots = str(directory_for_saving_plots) + "/"

# =========================================================
# Time related set-up; relevant for more complex models
# =========================================================
# In case one wants to modify dates compared to those set up in the configuration object / deverge from these setting
# if not, just comment out this whole part
start_date = "2006-03-30 00:00:00"
end_date = "2007-04-30 00:00:00"
spin_up_length = 365  # 365*3
start_date = pd.to_datetime(start_date)
end_date = pd.to_datetime(end_date)
# dict_with_dates_setup = {"start_date": start_date, "end_date": end_date, "spin_up_length":spin_up_length}
run_full_timespan = False
model.set_run_full_timespan(run_full_timespan)
model.set_start_date(start_date)
model.set_end_date(end_date)
model.set_spin_up_length(spin_up_length)
simulation_length = (model.end_date - model.start_date).days - model.spin_up_length
if simulation_length <= 0:
    simulation_length = 365
model.set_simulation_length(simulation_length)
model.set_date_ranges()
model.redo_input_and_measured_data_setup()

# Get to know some of the relevant time settings, read from a json configuration file
print(f"start_date: {model.start_date}")
print(f"start_date_predictions: {model.start_date_predictions}")
print(f"end_date: {model.end_date}")
print(
    f"full_data_range is {len(model.full_data_range)} "
    f"hours including spin_up_length of {model.spin_up_length} hours"
)
print(f"simulation_range is of length {len(model.simulation_range)} hours")

# Plot forcing data and observed streamflow
model._plot_input_data(read_measured_streamflow=True)

# Outer loop that goes over all date from the configuration json
list_of_dates_of_interest = list(
    pd.date_range(start=model.start_date_predictions, end=model.end_date, freq="1D")
)

configurationObject = model.configurationObject


# TODO: Sample Parameters
def likelihood_doc(y_model, y_observed):
    like_ctot = norm(loc=y_observed)
    logp = np.sum(like_ctot.logpdf(y_model))
    print("logp: " + str(logp))
    if np.isnan(logp):
        logp = -np.inf
    return logp


def likelihood_log_eq(y_model, y_observed):
    err_var = 5  # sigma**2
    n = len(y_observed)
    t1 = -(n / 2.0) * log(2 * pi)

    t2 = (n / 2) * log(err_var)

    t3 = 0
    for i in range(n):
        t3 += (y_model[i] - y_observed[i]) ** 2
    t3 *= 1 / (2 * err_var)
    return t1 - t2 - t3


def likelihood_eq(y_model, y_observed):
    err_var = 5  # sigma**2
    n = len(y_observed)
    prod = 1
    for i in range(n):
        t1 = 1 / ((2 * pi * err_var) ** 0.5)
        t2 = -((y_model[i] - y_observed[i]) ** 2) / (2 * err_var)
        t2 = exp(t2)
        term = t1 * t2
        prod = prod * term
    return prod


def likelihood_default(y_model, y_observed):
    """
    Computing Gaussian like likelihood, in case one has measured/observed data
    """

    res = calculate_gaussian_likelihood(y_observed, y_model)
    sum = 0
    for i in res:
        sum += log(i)
    return sum


# def simple_model(t, alpha, beta, l):
#     return l*np.exp(-alpha*t)*(np.cos(beta*t)+alpha/beta*np.sin(beta*t))


def likelihood(y_model, y_observed):
    # Change here
    return likelihood_log_eq(y_model, y_observed)


def likelihood_kernel(param_vec):
    _, y_model, y_observed, _ = run_model_single_parameter_node(model, param_vec)
    return likelihood(y_model, y_observed)


param_names = []
param_lower = []
param_upper = []
for param in configurationObject["parameters"]:
    # for now the Uniform distribution is only supported
    if param["distribution"] == "Uniform":
        param_names.append(param["name"])
        param_lower.append(param["lower"])
        param_upper.append(param["upper"])
    else:
        raise NotImplementedError(
            f"Sorry, the distribution {param['distribution']} is not supported yet"
        )
param_lower = np.array(param_lower)
param_upper = np.array(param_upper)

parameters_to_sample = SampledParam(
    tfp.distributions.Uniform, low=param_lower, high=param_upper
)

# The run_dream function expects a list rather than a single variable
sampled_parameter = [parameters_to_sample]

niterations = 10000
converged = False
total_iterations = niterations
nchains = 5

if __name__ == "__main__":
    start = time.time()
    sampled_params, log_ps = run_dream(
        sampled_parameter,
        likelihood_kernel,
        niterations=niterations,
        nchains=nchains,
        multitry=False,
        gamma_levels=4,
        adapt_gamma=True,
        history_thin=1,
        model_name="test_mcmc_hydrological",
        verbose=True,
    )
    end = time.time()
    print("Time needed: " + str(end - start))

    # for chain in range(len(sampled_params)):
    #    np.save('robertson_nopysb_dreamzs_5chain_sampled_params_chain_'+str(chain)+'_'+str(total_iterations), sampled_params[chain])
    #    np.save('robertson_nopysb_dreamzs_5chain_logps_chain_'+str(chain)+'_'+str(total_iterations), log_ps[chain])

    # Check convergence and continue sampling if not converged

    GR = Gelman_Rubin(sampled_params)
    print("At iteration: ", total_iterations, " GR = ", GR)
    # np.savetxt('robertson_nopysb_dreamzs_5chain_GelmanRubin_iteration_' + str(total_iterations) + '.txt', GR)

    old_samples = sampled_params
    # if np.any(GR > 1.2):
    #    starts = [sampled_params[chain][-1, :] for chain in range(nchains)]

    # while not converged:
    #     total_iterations += niterations

    #     sampled_params, log_ps = run_dream(sampled_parameter_names, likelihood, niterations=niterations,
    #                                        nchains=nchains, multitry=False, gamma_levels=4, adapt_gamma=True,
    #                                        history_thin=1, model_name='robertson_nopysb_dreamzs_5chain', verbose=True, restart=True)

    #     for chain in range(len(sampled_params)):
    #         np.save('robertson_nopysb_dreamzs_5chain_sampled_params_chain_' + str(chain) + '_' + str(total_iterations),
    #                     sampled_params[chain])
    #         np.save('robertson_nopysb_dreamzs_5chain_logps_chain_' + str(chain) + '_' + str(total_iterations),
    #                     log_ps[chain])

    #     old_samples = [np.concatenate((old_samples[chain], sampled_params[chain])) for chain in range(nchains)]
    #     GR = Gelman_Rubin(old_samples)
    #     print('At iteration: ', total_iterations, ' GR = ', GR)
    #     np.savetxt('robertson_nopysb_dreamzs_5chain_GelmanRubin_iteration_' + str(total_iterations)+'.txt', GR)

    #     if np.all(GR < 1.2):
    #         converged = True

    # Plot output
    total_iterations = len(old_samples[0])
    burnin = int(total_iterations / 2)
    samples = np.concatenate(
        (
            old_samples[0][burnin:, :],
            old_samples[1][burnin:, :],
            old_samples[2][burnin:, :],
            old_samples[3][burnin:, :],
            old_samples[4][burnin:, :],
        )
    )
    np.savetxt("samples_log_eq.out", samples, delimiter=",")

    ndims = len(old_samples[0][0])
    colors = sns.color_palette(n_colors=ndims)
    for dim in range(ndims):
        fig = plt.figure()
        sns.histplot(samples[:, dim], color=colors[dim], kde=True)
        fig.savefig("hydrological" + str(dim))

else:
    pass
    # run_kwargs = {'parameters':sampled_parameter_names, 'likelihood':likelihood, 'niterations':10000, 'nchains':nchains, 'multitry':False, 'gamma_levels':4, 'adapt_gamma':True, 'history_thin':1, 'model_name':'robertson_nopysb_dreamzs_5chain', 'verbose':True}
