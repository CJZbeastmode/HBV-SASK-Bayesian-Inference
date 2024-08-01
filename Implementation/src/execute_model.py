TIME_COLUMN_NAME = "TimeStamp"
INDEX_COLUMN_NAME = "Index_run"
PLOT_FORCING_DATA = True
PLOT_ALL_THE_RUNS = True
QOI_COLUMN_NAME = "model"  # "Value"
MODEL = "hbv-sask"  #'hbv-sask' # 'banchmark_model' or 'simple_model' or 'hbv-sask'
QOI_COLUMN_NAME = "Q_cms"
QOI_COLUM_NAME_MESURED = "streamflow"


def run_model_single_parameter_node(
    model,
    parameter_value_dict,
    unique_index_model_run=0,
    qoi_column_name=QOI_COLUMN_NAME,
    qoi_column_name_measured=QOI_COLUM_NAME_MESURED,
    **kwargs
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
    else:
        y_t_observed = None
    return unique_index_model_run, y_t_model, y_t_observed, parameter_value_dict
