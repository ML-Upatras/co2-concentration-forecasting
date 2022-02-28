# DATASET PARAMETERS
from darts.utils.likelihood_models import GaussianLikelihood
from darts.utils.utils import SeasonalityMode
from torch.nn import MSELoss

SAMPLE_RATE = "60T"
SPLIT_PERCENTAGE = 0.8

# MODEL PARAMS
INPUT_CHUNK_LENGTH = 1 * 24 * 10
FORECAST_HORIZON = 1 * 24 * 7

# TRAINING PARAMS
EPOCHS = 20
BATCH_SIZE = 16

# DARTS PARAMS
FORCE_RESET = True

# RETRAINING
RETRAIN = True

# HISTORICAL FORECAST KWARGS
historical_forecast_kwargs = dict(
    forecast_horizon=FORECAST_HORIZON,
    stride=1,
    last_points_only=True,
    # Whether the returned forecasts can go beyond the series' end or not
    # (This make problem with future covariates so keep it false)
    overlap_end=False,
    verbose=True,
)

# TFT KWARGS
tft_kwargs = dict(
    # forecasting parameters
    input_chunk_length=INPUT_CHUNK_LENGTH,
    output_chunk_length=FORECAST_HORIZON,

    # hyperparameters
    hidden_size=64,
    lstm_layers=1,
    num_attention_heads=4,
    dropout=0.1,
    batch_size=BATCH_SIZE,

    # darts parameters
    add_relative_index=True,
    add_encoders=None,

    # training parameters
    n_epochs=EPOCHS,
    likelihood=None,
    loss_fn=MSELoss(),
    random_state=42,

    # logging parameters
    model_name=f"tft_{FORECAST_HORIZON}_retrain_{RETRAIN}",
    save_checkpoints=True,
    log_tensorboard=True,

    # reset
    force_reset=FORCE_RESET
)

# RNN KWARGS
lstm_kwargs = dict(
    # forecasting parameters
    input_chunk_length=INPUT_CHUNK_LENGTH,
    output_chunk_length=FORECAST_HORIZON,

    # hyperparameters
    model="LSTM",
    hidden_size=16,
    n_rnn_layers=2,
    dropout=0.1,
    batch_size=BATCH_SIZE,
    optimizer_kwargs={"lr": 1e-3},

    # training parameters
    n_epochs=EPOCHS,
    loss_fn=MSELoss(),
    random_state=42,

    # logging parameters
    model_name=f"lstm_{FORECAST_HORIZON}_retrain_{RETRAIN}",
    save_checkpoints=True,
    log_tensorboard=True,

    # reset
    force_reset=FORCE_RESET
)

# NBEATS KWARGS
nbeats_kwargs = dict(
    # forecasting parameters
    input_chunk_length=INPUT_CHUNK_LENGTH,
    output_chunk_length=FORECAST_HORIZON,

    # hyperparameters
    generic_architecture=True,
    num_stacks=10,
    num_blocks=1,
    num_layers=4,
    layer_widths=512,
    batch_size=BATCH_SIZE,

    # training parameters
    n_epochs=EPOCHS,
    loss_fn=MSELoss(),
    random_state=42,

    # logging parameters
    model_name=f"nbeats_{FORECAST_HORIZON}_retrain_{RETRAIN}",
    save_checkpoints=True,
    log_tensorboard=True,

    # reset
    force_reset=FORCE_RESET
)

# DEEPAR
deepar_kwargs = dict(
    # forecasting parameters
    input_chunk_length=INPUT_CHUNK_LENGTH,
    training_length=FORECAST_HORIZON,

    # hyperparameters
    model="LSTM",
    hidden_dim=20,
    dropout=0,
    batch_size=BATCH_SIZE,
    optimizer_kwargs={"lr": 1e-3},

    # training parameters
    n_epochs=EPOCHS,
    likelihood=GaussianLikelihood(),
    random_state=42,

    # logging parameters
    model_name=f"deepar_{FORECAST_HORIZON}_retrain_{RETRAIN}",
    save_checkpoints=True,
    log_tensorboard=True,

    # reset
    force_reset=FORCE_RESET
)

# TCN KWARGS
tcn_kwargs = dict(
    # forecasting parameters
    input_chunk_length=INPUT_CHUNK_LENGTH,
    output_chunk_length=FORECAST_HORIZON,

    # hyperparameters
    dropout=0.1,
    dilation_base=2,
    weight_norm=True,
    kernel_size=5,
    num_filters=3,
    batch_size=BATCH_SIZE,

    # training parameters
    n_epochs=EPOCHS,
    loss_fn=MSELoss(),
    random_state=42,

    # logging parameters
    model_name=f"tcn_{FORECAST_HORIZON}_retrain_{RETRAIN}",
    save_checkpoints=True,
    log_tensorboard=True,

    # reset
    force_reset=FORCE_RESET
)

# TRANSFORMER KWARGS
transformer_kwargs = dict(
    # forecasting parameters
    input_chunk_length=INPUT_CHUNK_LENGTH,
    output_chunk_length=FORECAST_HORIZON,

    # hyperparameters
    d_model=16,
    nhead=8,
    num_encoder_layers=2,
    num_decoder_layers=2,
    dim_feedforward=128,
    dropout=0.1,
    activation="relu",
    batch_size=BATCH_SIZE,

    # training parameters
    n_epochs=EPOCHS,
    loss_fn=MSELoss(),
    random_state=42,

    # logging parameters
    model_name=f"transformer_{FORECAST_HORIZON}_retrain_{RETRAIN}",
    save_checkpoints=True,
    log_tensorboard=True,

    # reset
    force_reset=FORCE_RESET
)

# ARIMA KWARGS
arima_kwargs = dict()

# EXPONENTIAL SMOOTHING KWARGS
exp_smoothing_kwargs = dict(
    seasonal_periods=12
)

# THETA KWARGS
theta_kwargs = dict(
    season_mode=SeasonalityMode.NONE
)

# FFT KWARGS
fft_kwargs = dict()
