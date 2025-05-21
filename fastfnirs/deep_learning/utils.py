import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from skorch.callbacks import Callback
from torch.utils.data import TensorDataset
from skorch.dataset import Dataset, ValidSplit


def to_torch_dataset(X, y):
    return TensorDataset(torch.from_numpy(X), torch.from_numpy(y))


class ValScoring(Callback):
    """
    A way to score the model on the validation set at the end of each epoch. Requires a hacky way to access the pipeline. See ValSplitter for a better way.
    """

    def __init__(self, X_val, y_val, scorer, name="val_score"):
        self.X_val = X_val
        self.y_val = y_val
        self.scorer = scorer
        self.name = name

    def on_epoch_end(self, net, **kwargs):
        pred_y = net.pipeline.predict(self.X_val)
        score = self.scorer(pred_y, self.y_val)
        net.history.record(self.name, score)


class ValSplitter(ValidSplit):
    """
    A way to split the dataset into training and validation sets. Doesn't apply pipeline to the validation set.
    """

    def __init__(self, warn=True, **kwargs):
        print("WARNING: ValSplitter doesn't apply pipeline to the validation set.")
        self.validation_dataset = kwargs.pop("validation_dataset")
        super().__init__(**kwargs)

    def __call__(self, dataset, y=None, groups=None):
        return dataset, self.validation_dataset


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class PlotLayer(nn.Module):
    def __init__(self, name="", type="time"):
        super().__init__()
        self.name = name
        self.type = type

    def plot_time(self, x, sample_ix=0):
        print(x.shape)
        s, ic, c, t = x.shape
        fmax = 4
        f = min(ic, fmax)
        fig, axs = plt.subplots(f, 1, figsize=(5, 2 * f))
        for filter_ix in range(f):
            ch_ix = 0
            z = x[sample_ix, filter_ix, ch_ix, :].detach().cpu().numpy()
            # vmax = np.quantile(np.abs(z), 0.9)
            # vmin = -vmax
            ax = axs[filter_ix] if f > 1 else axs
            ax.plot(z)
            ax.text(
                -0.15,
                0.5,
                f"Filter {filter_ix}",
                fontsize=12,
                ha="center",
                va="center",
                rotation="vertical",
                transform=ax.transAxes,
            )
        plt.tight_layout()
        plt.suptitle(f"{self.name}", y=1.05)
        # plt.savefig(f'fig/time_{self.name}.png', bbox_inches='tight', dpi=300)
        plt.show()

    def plot_time_multi(self, x, sample_ix=0):
        s, ic, c, t = x.shape
        print(f"{s=}, {ic=}, {c=}, {t=}")

        # number of filters to plot
        fmax = 4
        f = min(ic, fmax)

        n_ch_to_plot = 4
        if c > n_ch_to_plot:
            ch_ixs = np.linspace(0, c - 1, n_ch_to_plot).astype(int)
        else:
            ch_ixs = np.arange(c)
        print(f"{ch_ixs=}")

        fw = 2 + np.round(t / 60) * 1
        fig, axs = plt.subplots(f, 1, figsize=(fw, 2 * f))

        for filter_ix in range(f):
            xf = x[sample_ix, filter_ix, ch_ixs, :].detach().cpu().numpy()
            print(f"{xf.shape=}")

            ax = axs[filter_ix] if f > 1 else axs
            ax = plot_chs(
                xf,
                1,
                scale="auto",
                max_amplitude=0.18,
                ax=ax,
            )
            ax.text(
                -0.15,
                0.5,
                f"Filter {filter_ix}",
                fontsize=12,
                ha="center",
                va="center",
                rotation="vertical",
                transform=ax.transAxes,
            )
        plt.tight_layout()
        plt.suptitle(f"{self.name}", y=1.05)
        # plt.savefig(
        #     f"fig/eegnet/time_multi_{self.name}.png", bbox_inches="tight", dpi=300
        # )
        plt.show()

    def forward(self, x):
        if self.type == "time":
            self.plot_time(x)
        elif self.type == "time_multi":
            self.plot_time_multi(x)
        return x


def plot_chs(data, sampling_rate, scale=1, max_amplitude=1, ax=None):
    """
    Plots multi-channel data.

    Parameters:
    data (np.ndarray): 2D array where each row is a time series for one channel.
    channel_names (list): List of channel names.
    sampling_rate (int): Sampling rate of the data.
    scale (float): Scaling factor for the amplitude of the signals.
    max_amplitude (float): Maximum amplitude allowed for each channel to prevent overflow.
    """
    data = np.array(data)
    num_channels, num_samples = data.shape
    time = np.arange(num_samples) / sampling_rate

    # make everything start at 0
    data = data - data[:, 0][:, None]

    if ax is None:
        fig, ax = plt.subplots(figsize=((1 / 20) * len(time), num_channels * 0.2))

    offset = 0
    offset_step = 0.2

    for i in range(num_channels):
        if scale == "auto":
            channel_data = data[i] / np.max(np.abs(data[i])) * max_amplitude
        else:
            channel_data = scale * data[i]
        # Mask the data that exceeds the max_amplitude
        channel_data = np.ma.masked_outside(channel_data, -max_amplitude, max_amplitude)
        ax.plot(
            time,
            channel_data + offset,
            # label=channel_names[i],
            linewidth=1,
            color="black",
        )

        offset += (
            offset_step  # Add a fixed offset for each channel to stack them vertically
        )

    # ax.set_xlabel('Time (s)')
    # ax.set_yticks(np.arange(0, num_channels * offset_step, offset_step))
    # ax.set_yticklabels(channel_names)
    ax.set_yticks([])
    ax.set_xticks([])
    # ax.set_ylim(-max_amplitude-.1, num_channels * offset_step-.1)
    return ax
