import pandas as pd
from darts.metrics import mse, mae, rmse, r2_score


def evaluate_model(ts, hf_scaled, scaler, model_name, fh, save_path):
    # set results filename
    RESULTS_FILE = save_path / f"results_fh-{fh}.csv"

    # unscale historical forecast (ts is already unscaled)
    hf = scaler.inverse_transform(hf_scaled)

    # calculate metrics
    mean_squared_error = mse(ts, hf)
    mean_absolute_error = mae(ts, hf)
    root_mean_squared_error = rmse(ts, hf)
    r2_score_ = r2_score(ts, hf)

    # round to 4 decimal places
    mean_absolute_error = round(mean_absolute_error, 4)
    mean_squared_error = round(mean_squared_error, 4)
    root_mean_squared_error = round(root_mean_squared_error, 4)
    r2_score_ = round(r2_score_, 4)

    # if csv exists open it
    if RESULTS_FILE.exists():
        df = pd.read_csv(RESULTS_FILE)
    else:
        df = pd.DataFrame(columns=["model_name",
                                   "mean_squared_error",
                                   "mean_absolute_error",
                                   "root_mean_squared_error",
                                   "r2_score_"])

    # append new row
    df = df.append({"model_name": model_name,
                    "mean_squared_error": mean_squared_error,
                    "mean_absolute_error": mean_absolute_error,
                    "root_mean_squared_error": root_mean_squared_error,
                    "r2_score_": r2_score_}, ignore_index=True)
    df = df.drop_duplicates(subset="model_name", keep="last")

    # save to csv
    df.to_csv(RESULTS_FILE, index=False)
