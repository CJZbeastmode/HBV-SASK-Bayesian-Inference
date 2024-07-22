# Bachelorarbeit
This repository contains the bachelor thesis of Chengjie Zhou (Jay) from Technical University of Munich.

## Installation
### Setting-up the conda environment:
The easiest way is just to run the follwing command form the terminal once you have a conda installed:
```bash
conda create -n hbv_uq_env python=3.11 --file requirements.txt
```

## Thesis
Execute the following command to compile the LATEX file
```bash
latexmk -pdf main.tex
```

## Usage

### Run Algorithm
Configure the run_config file and run  
```bash
python Implementation/src/run.py
```

### Visualization
After storing the data in the root, run the Jupyter notebook from Implementation/src/visualization.ipynb. Before running the notebook, make sure to fill out the viz_config.json file on the root level correspondingly.

### Config File
#### Run Config
* configPath (required): the config file that is used for the model
* basis (required): the basis of the data that is used for the model
* mode (required): the mode of the algorithm, options: mh, parallel_mh, gpmh, dream
* separate_chains (optional, default=false): to determine whether the output file is supposed to record the data separately by chains, only relevant for algorithms that sample data using multiple chains, including parallel_mh and dream
* burnin_fac (optional, default=5): the burn in factor that is used for the result of the MCMC algorithm. The first 1/burnin_fac percentage of the entire data is going to be discarded
* effective_sample_size (optional, default=1): only the every n_th data is going to be collected. Default 1: no data point is going to be discarded
* output_file_name (optional, default="mcmc_data.out"): the file name of the saved output result
* kwargs (optional): a dictionary in form of JSON that is used for specific algorithm input parameters

#### Viz Config
* configPath (required): the config file that is used for the model
* basis (required): the basis of the data that is used for the model
* input_file (required): the data in the input file. It could be seperately recorded or merged
* sep_viz (optional, default=False): the option to visualize the data by chains. If false, then the entire dataframe is going to be visualized. If true, different chains are going to be visualized individually, before a comparison visualization is going to be given
* monte_carlo_repetition (optional, default=1000): the number of iterations for the monte carlo method for the comparison of the Bayesian inference result

#### MH
* version (optional, default="ignoring"): version of the MH algorithm. Options: ignoring, refl_bound, aggr
* sd_transition_factor (optional, default=6): the standard deviation factor of the transition kernel. The standard deviation is given by (upper bound - lower bound) / sd_transition_factor
* likelihood_sd (optional, default=1): the standard deviation parameter for independent likelihood function, or the standard deviation parameter factor for dependent likelihood function (standard deviation: likelihood_sd * y_error).
* likelihood_dependence (optional, required if likelihood_sd is present, default=False): to select whether to use the dependent likelihood function or the independent likelihood function
* max_probability (optional, default=False): the acceptance rate will take the maximum probability value of the acceptance rate array if set true, otherwise the mean
* iterations (optional, default=10000): number of iterations
* init_method (optional, default="random"): specify the starting state of the Dream MCMC algorithm. Options: random, min, max, q1_prior, mean_prior, q3_prior, q1_posterior, median_posterior, q3_posterior

#### Parallel_MH
* version (optional, default="ignoring"): version of the MH algorithm. Options: ignoring, refl_bound, aggr
* chains (optional, default=4): number of chains
* sd_transition_factor (optional, default=6): the standard deviation factor of the transition kernel. The standard deviation is given by (upper bound - lower bound) / sd_transition_factor
* likelihood_sd (optional, default=1): the standard deviation parameter for independent likelihood function, or the standard deviation parameter factor for dependent likelihood function (standard deviation: likelihood_sd * y_error).
* likelihood_dependence (optional, required if likelihood_sd is present, default=False): to select whether to use the dependent likelihood function or the independent likelihood function
* max_probability (optional, default=False): the acceptance rate will take the maximum probability value of the acceptance rate array if set true, otherwise the mean
* iterations (optional, default=2500): number of iterations
* init_method (optional, default="random"): specify the starting state of the Dream MCMC algorithm. Options: random, min, max, q1_prior, mean_prior, q3_prior, q1_posterior, median_posterior, q3_posterior

#### GPMH
* num_proposals (optional, default=8): the numbers of proposal points in each iteration
* num_accepted (optional, default=4): the numbers of accepted points in each iteration
* likelihood_sd (optional, default=1): the standard deviation parameter for independent likelihood function, or the standard deviation parameter factor for dependent likelihood function (standard deviation: likelihood_sd * y_error).
* likelihood_dependence (optional, required if likelihood_sd is present, default=False): to select whether to use the dependent likelihood function or the independent likelihood function
* sd_transition_factor (optional, default=6): the standard deviation factor of the transition kernel. The standard deviation is given by (upper bound - lower bound) / sd_transition_factor
* version (optional, default="ignoring"): version of the MH algorithm. Options: ignoring, refl_bound, aggr
* iterations (optional, default=2500): number of iterations
* init_method (optional, default="random"): specify the starting state of the Dream MCMC algorithm. Options: random, min, max, q1_prior, mean_prior, q3_prior, q1_posterior, median_posterior, q3_posterior


#### Dream
* iterations (optional, default=1250): number of iterations
* chains (optional, default=8): number of chains
* DEpairs (optional, default=1)
* multitry (optional, default=False)
* hardboundaries (optional, default=True)
* crossover_burnin (optional, default=0)
* nCR (optional, default=3)
* snooker (optional, default=0)
* p_gamma_unity (optional, default=0)
* init_method (optional, default="random"): specify the starting state of the Dream MCMC algorithm. Options: random, min, max, q1_prior, mean_prior, q3_prior, q1_posterior, median_posterior, q3_posterior
* likelihood_sd (optional, default=1): the standard deviation parameter for independent likelihood function, or the standard deviation parameter factor for dependent likelihood function (standard deviation: likelihood_sd * y_error).
* likelihood_dependence (optional, required if likelihood_sd is present, default=False): to select whether to use the dependent likelihood function or the independent likelihood function  

https://pydream.readthedocs.io/en/latest/pydream.html


## License
