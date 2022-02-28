from matplotlib import pyplot as plt


def plot_hf(ts, hf_scaled, model_name, scaler, fh, save_path):
    # set up plot
    plt.figure(figsize=(32, 8))
    plt.title(f"{model_name} FH-{fh}")

    # transform historical forecast (ts is already unscaled)
    hf = scaler.inverse_transform(hf_scaled)

    # plot
    ts.plot(label=f"Real")
    hf.plot(label=f"{model_name} FH-{fh} Predicted")
    plt.legend()

    # save
    FIGURE_PATH = save_path / f"{model_name}_fh-{fh}.png"
    plt.savefig(FIGURE_PATH)
