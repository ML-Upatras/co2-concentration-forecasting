import pandas as pd
from pathlib import Path
from darts import TimeSeries
from darts.metrics import rmse, mae

from stac import friedman_test


def calc_avg(split, fh):
    HF_NO_RETRAIN_DATA_PATH = Path(f"{split}/results/retrain_False/hf_{fh}.csv")
    HF_RETRAIN_DATA_PATH = Path(f"{split}/results/retrain_True/hf_{fh}.csv")

    # load data
    hf_no_retrain_df = pd.read_csv(HF_NO_RETRAIN_DATA_PATH)
    hf_retrain_df = pd.read_csv(HF_RETRAIN_DATA_PATH)

    # keep avg and real
    avg_no_retrain = TimeSeries.from_series(hf_no_retrain_df[["tft", "arima"]].mean(axis=1))
    real_no_retrain = TimeSeries.from_series(hf_no_retrain_df["real"])

    avg_retrain = TimeSeries.from_series(hf_retrain_df[["tft", "arima"]].mean(axis=1))
    real_retrain = TimeSeries.from_series(hf_retrain_df["real"])

    # calculate metrics
    mean_absolute_error_no_retrain = mae(real_no_retrain, avg_no_retrain)
    root_mean_squared_error_no_retrain = rmse(real_no_retrain, avg_no_retrain)

    mean_absolute_error_retrain = mae(real_retrain, avg_retrain)
    root_mean_squared_error_retrain = rmse(real_retrain, avg_retrain)

    # add to dataframe
    retrain_row = {
        "model_name": "avg",
        "mean_absolute_error": mean_absolute_error_retrain,
        "root_mean_squared_error": root_mean_squared_error_retrain,
        "fh": fh,
        "split": split,
        "retrain": "yes"
    }
    avg_retrain_df = pd.DataFrame(retrain_row, index=[0])

    no_retrain_row = {
        "model_name": "avg",
        "mean_absolute_error": mean_absolute_error_no_retrain,
        "root_mean_squared_error": root_mean_squared_error_no_retrain,
        "fh": fh,
        "split": split,
        "retrain": "no"
    }
    avg_no_retrain_df = pd.DataFrame(no_retrain_row, index=[0])

    return avg_retrain_df, avg_no_retrain_df


if __name__ == '__main__':
    results_list = []

    for fh, split in [(1, "80-20"), (24, "80-20"), (168, "80-20"), (1, "90-10"), (24, "90-10")]:
        NO_RETRAIN_DATA_PATH = Path(f"{split}/results/retrain_False/results_fh-{fh}.csv")
        RETRAIN_DATA_PATH = Path(f"{split}/results/retrain_True/results_fh-{fh}.csv")

        # load data
        no_retrain_df = pd.read_csv(NO_RETRAIN_DATA_PATH)
        retrain_df = pd.read_csv(RETRAIN_DATA_PATH)

        # drop unnecessary columns
        col_to_drop = ["mean_squared_error", "r2_score_"]
        no_retrain_df = no_retrain_df.drop(columns=col_to_drop)
        retrain_df = retrain_df.drop(columns=col_to_drop)

        # set forecast horizon
        no_retrain_df["fh"] = fh
        retrain_df["fh"] = fh

        # split
        no_retrain_df["split"] = split
        retrain_df["split"] = split

        # set retrain
        no_retrain_df["retrain"] = "no"
        retrain_df["retrain"] = "yes"

        # calculate avg
        avg_retrain_df, avg_no_retrain_df = calc_avg(split, fh)

        # concat to df
        results_list.append(avg_retrain_df)
        results_list.append(avg_no_retrain_df)

        # sort dataframes by mean_absolute_error
        no_retrain_df = no_retrain_df.sort_values(by=['mean_absolute_error']).reset_index(drop=True)
        retrain_df = retrain_df.sort_values(by=['mean_absolute_error']).reset_index(drop=True)

        # set the model names that are retrained
        retrain_models = ["tft", "nbeats", "deepar", "lstm", "transformer", "tcn"]

        # append to results list
        results_list.append(no_retrain_df)
        results_list.append(retrain_df)

    # concatenate results
    results_df = pd.concat(results_list)
    results_df = results_df.reset_index(drop=True)

# export tables
for fh, split in [(1, "80-20"), (24, "80-20"), (168, "80-20"), (1, "90-10"), (24, "90-10")]:
    split_condition = results_df["split"] == split
    fh_condition = results_df["fh"] == fh
    specific_df = results_df[split_condition & fh_condition]
    specific_df = specific_df.reset_index(drop=True)

    specific_df = specific_df.sort_values(by=["model_name", "mean_absolute_error"])
    specific_df.to_csv(f"results/table-{split}-{fh}.csv")
print("Tables exported!")

# prepare friedman dataset
for metric in ["mean_absolute_error", "root_mean_squared_error"]:
    friedman_df = pd.DataFrame(
        columns=['avg_retrain_no',
                 'tft_retrain_no',
                 'fft_retrain_no',
                 'transformer_retrain_no',
                 'arima_retrain_no',
                 'nbeats_retrain_no',
                 'tcn_retrain_no',
                 'naive_retrain_no',
                 'theta_retrain_no',
                 'exponential_smoothing_retrain_no',
                 'lstm_retrain_no',
                 'deepar_retrain_no',
                 'deepar_retrain_yes',
                 'fft_retrain_yes',
                 'tft_retrain_yes',
                 'arima_retrain_yes',
                 'naive_retrain_yes',
                 'theta_retrain_yes',
                 'nbeats_retrain_yes',
                 'exponential_smoothing_retrain_yes',
                 'transformer_retrain_yes',
                 'lstm_retrain_yes',
                 'tcn_retrain_yes']
    )
    for fh, split in [(1, "80-20"), (24, "80-20"), (168, "80-20"), (1, "90-10"), (24, "90-10")]:
        split_condition = results_df["split"] == split
        fh_condition = results_df["fh"] == fh
        specific_df = results_df[split_condition & fh_condition]
        specific_df = specific_df.reset_index(drop=True)

        specific_df["model_name"] = specific_df["model_name"] + "_retrain_" + specific_df["retrain"]
        specific_df = specific_df.set_index("model_name")[[metric]].T

        friedman_df = friedman_df.append(specific_df)
        friedman_df = friedman_df.reset_index(drop=True)

    # actual friedman
    statistic, p_value, ranking, rank_cmp = friedman_test(*friedman_df.to_dict().values())
    friedman = pd.DataFrame(index=friedman_df.columns.tolist())
    friedman['ranking'] = ranking
    friedman.sort_values(by='ranking').to_csv(f"results/{metric}_friedman.csv")
    print("Friedman finished!")
