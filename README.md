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

### Relevant Files/Scripts:
* `running_hbv_model.py`  - Demonstrates a simple example of how to run the HBV-SASK model for a single set of parameters and/or states, sequentially over data.

### Json Configuration File:
* `time_settings` 
* `model_settings` 
* `model_paths`
* `simulation_settings`  
* `parameters` - list of dictionaries; for each uncertain parameter, there is a dictionary that stores its: name, distribution, parameters of the distribution (e.g., for Uniform distribution - lower upper), a default value
* `states` - list of dictionaries; for each uncertain state

## License
