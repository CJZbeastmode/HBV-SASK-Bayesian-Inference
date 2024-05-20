import sys
import pathlib

sys.path.append('/Users/jay/Desktop/Bachelorarbeit')
from Implementation.dependencies.hbv_sask.model import HBVSASKModel as hbvmodel

hbv_model_data_path = pathlib.Path("/Users/jay/Desktop/Bachelorarbeit/Implementation/dependencies/hbv_sask/data")
inputModelDir = hbv_model_data_path

def get_model(configurationPath, basis):
    model = hbvmodel.HBVSASKModel(
        configurationObject=pathlib.Path(configurationPath),
        inputModelDir=inputModelDir,
        workingDir=hbv_model_data_path / basis / "model_runs" / "running_the_model_parallel_simple",
        basis=basis,
        writing_results_to_a_file=False,
        plotting=False
    )
    print(f"start_date: {model.start_date}")
    print(f"start_date_predictions: {model.start_date_predictions}")
    print(f"end_date: {model.end_date}")
    print(f"simulation length: {model.simulation_length}")
    print(f"full_data_range is {len(model.full_data_range)} "
        f"hours including spin_up_length of {model.spin_up_length} hours")
    print(f"simulation_range is of length {len(model.simulation_range)} hours")
    return model


"""
start_date = '2006-03-30 00:00:00'
end_date = '2007-04-30 00:00:00'
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
"""
