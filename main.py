from pathlib import Path

from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.models import NaiveSeasonal, TFTModel, BlockRNNModel, NBEATSModel, RNNModel, TCNModel, TransformerModel, \
    ARIMA, ExponentialSmoothing, Theta, FFT
from matplotlib import pyplot as plt

import src.config as cfg
from src.utils import evaluate_model
from src.dataset import create_future_covariates, load_co2_data
from src.visualizations import plot_hf

FIGURES_PATH = Path("figures")
FIGURES_PATH.mkdir(exist_ok=True, parents=True)

RESULTS_PATH = Path("results")
RESULTS_PATH.mkdir(exist_ok=True, parents=True)

if __name__ == '__main__':
    # load and prepare data
    df = load_co2_data(sample_rate=cfg.SAMPLE_RATE)

    # load future covariates
    future_cov = create_future_covariates(df["timestamp"])
    df = df.drop(columns=["timestamp"])

    # convert to TimeSeries Dataset
    dataset_ts = TimeSeries.from_dataframe(df)

    # isolate series
    target_ts = dataset_ts["co2"]
    past_cov = dataset_ts[["temperature", "humidity", "light_intensity"]]

    # SPLIT
    print("\nSplitting data...")
    train_ts, test_ts = target_ts.split_before(cfg.SPLIT_PERCENTAGE)
    train_past_cov_ts, test_past_cov_ts = past_cov.split_before(cfg.SPLIT_PERCENTAGE)
    train_future_cov_ts, test_future_cov_ts = future_cov.split_before(cfg.SPLIT_PERCENTAGE)

    # assert that all the train and test splits are equal length
    assert len(train_ts) == len(train_past_cov_ts) == len(train_future_cov_ts)
    assert len(test_ts) == len(test_past_cov_ts) == len(test_future_cov_ts)

    print(f"\nTrain length: {len(train_ts)}")
    print(f"Test length: {len(test_ts)}")

    # SCALING
    print("\nScaling data...")
    scaler = Scaler()
    train_ts = scaler.fit_transform(train_ts)
    target_ts = scaler.transform(target_ts)

    past_cov_scaler = Scaler()
    train_past_cov_ts = past_cov_scaler.fit_transform(train_past_cov_ts)
    past_cov = past_cov_scaler.transform(past_cov)

    future_co_scaler = Scaler()
    train_future_cov_ts = future_co_scaler.fit_transform(train_future_cov_ts)
    future_cov = future_co_scaler.transform(future_cov)

    # set up models
    models = {
        "naive": NaiveSeasonal(K=1),
        "tft": TFTModel(**cfg.tft_kwargs),
        "lstm": BlockRNNModel(**cfg.lstm_kwargs),
        "nbeats": NBEATSModel(**cfg.nbeats_kwargs),
        "deepar": RNNModel(**cfg.deepar_kwargs),
        "tcn": TCNModel(**cfg.tcn_kwargs),
        "transformer": TransformerModel(**cfg.transformer_kwargs),
        "arima": ARIMA(**cfg.arima_kwargs),
        "exponential_smoothing": ExponentialSmoothing(**cfg.exp_smoothing_kwargs),
        "theta": Theta(**cfg.theta_kwargs),
        "fft": FFT(**cfg.fft_kwargs)
    }

    # initialize for save forecasts
    historical_forecasts = {}

    for model_name, model in models.items():
        print(f"\nFitting {model_name} model...")

        # fit model
        if model_name in ["naive", "arima", "exponential_smoothing", "theta", "fft"]:
            model.fit(series=train_ts)
        elif model_name in ["tft"]:
            model.fit(
                series=train_ts,
                past_covariates=train_past_cov_ts,
                future_covariates=train_future_cov_ts,
                verbose=True
            )
        elif model_name in ["lstm", "nbeats", "tcn", "transformer"]:
            model.fit(
                series=train_ts,
                past_covariates=train_past_cov_ts,
                verbose=True
            )
        elif model_name in ["deepar"]:
            model.fit(
                series=train_ts,
                future_covariates=train_future_cov_ts,
                verbose=True
            )
        else:
            raise NotImplementedError

        # update kwargs (copy is essential because otherwise the original kwargs are modified)
        hf_kwargs = cfg.historical_forecast_kwargs.copy()
        hf_kwargs["series"] = target_ts
        hf_kwargs["start"] = train_ts.end_time()

        # add past cogitates
        if model_name in ["tft", "lstm", "nbeats", "tcn", "transformer"]:
            print("Add past covariates to historical forecast kwargs...")
            hf_kwargs["past_covariates"] = past_cov
            hf_kwargs["retrain"] = cfg.RETRAIN
        else:
            print("No past covariates")

        # add future covariates
        if model_name in ["tft", "deepar"]:
            print("Add future covariates to historical forecast kwargs...")
            hf_kwargs["future_covariates"] = future_cov
            hf_kwargs["retrain"] = cfg.RETRAIN
        else:
            print("No future covariates")

        test_hf = model.historical_forecasts(**hf_kwargs)
        historical_forecasts[model_name] = test_hf

        # plot the forecasts
        plot_hf(
            ts=test_ts,
            hf_scaled=test_hf,
            scaler=scaler,
            model_name=model_name,
            fh=cfg.FORECAST_HORIZON,
            save_path=FIGURES_PATH
        )
        # keep the results
        evaluate_model(
            ts=test_ts,
            hf_scaled=test_hf,
            scaler=scaler,
            model_name=model_name,
            fh=cfg.FORECAST_HORIZON,
            save_path=RESULTS_PATH
        )

    # plot all the forecasts from historical forecasts dict
    plt.figure(figsize=(34, 8))
    for model_name, hf in historical_forecasts.items():
        hf.plot(label=model_name)
        plt.savefig(FIGURES_PATH / f"{cfg.FORECAST_HORIZON}_all.png")