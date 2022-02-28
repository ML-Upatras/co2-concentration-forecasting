# co2-forecasting-benchmark

The code for the paper: "Forecasting CO2 emissions from a single-sensor."

#### Requirements

Install darts with conda:

```jupyter
conda install -c conda-forge -c pytorch u8darts-all
```

#### Setup Experiment details

Modify the config.py into the src/ directory to change the experiment details.

- **SAMPLE_RATE:** Sample rate of the data.
- **SPLIT_PERCENTAGE:** Percentage of the data to use for training.
- **INPUT_CHUNK_LENGTH:** Length of the past input records that are fed into the model.
- **FORECAST_HORIZON:** Number of future records to predict.
- **EPOCHS:** Number of epochs to train for.
- **BATCH_SIZE:** Batch size to use for training.
- **FORCE_RESET:** If True, the model will overwrite the previous trained model with the same name.
- **RETRAIN:** If True, the model will retrain from scratch for each new evaluation data point.

#### Execution

```jupyter
python main.py
```

#### Clear environment from results

```jupyter
python restart.py
```