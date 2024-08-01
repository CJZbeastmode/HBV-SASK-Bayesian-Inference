import sys
import pathlib

# TODO: Change root path
root = "/Users/jay/Desktop/Bachelorarbeit"

sys.path.append(root)
from Implementation.dependencies.hbv_sask.model import HBVSASKModel as hbvmodel

hbv_model_data_path = pathlib.Path(f"{root}/Implementation/dependencies/hbv_sask/data")
inputModelDir = hbv_model_data_path


def get_model(configurationPath, basis):
    model = hbvmodel.HBVSASKModel(
        configurationObject=pathlib.Path(configurationPath),
        inputModelDir=inputModelDir,
        workingDir=hbv_model_data_path
        / basis
        / "model_runs"
        / "running_the_model_parallel_simple",
        basis=basis,
        writing_results_to_a_file=False,
        plotting=False,
    )
    print(f"start_date: {model.start_date}")
    print(f"start_date_predictions: {model.start_date_predictions}")
    print(f"end_date: {model.end_date}")
    print(f"simulation length: {model.simulation_length}")
    print(
        f"full_data_range is {len(model.full_data_range)} "
        f"hours including spin_up_length of {model.spin_up_length} hours"
    )
    print(f"simulation_range is of length {len(model.simulation_range)} hours")
    return model
