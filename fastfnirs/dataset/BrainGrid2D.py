"""
2D grid for mapping locations to grid cells
"""

# %%
#!%load_ext autoreload
#!%autoreload 2

import numpy as np
import matplotlib.pyplot as plt
from math import sin, cos
import mne

from fastfnirs.dataset.BrainDataset import BrainDataset
from fastfnirs.utils import get_all_subjects, get_cwd, load_from

dataset_name = "nemo"
subjects = get_all_subjects(dataset_name)
subject = subjects[0]
bd = BrainDataset.load_data_2d(dataset_name, subjects=subjects)
ee = bd.epochs_dict[subject]
chs = [ch for ch in ee.info["chs"] if "hbo" in ch["ch_name"]]


class BrainGrid:
    def __init__(
        self,
        w=11,
        h=5,
        x_min=-0.095,
        x_max=0.095,
        y_min=-0.03,
        y_max=0.095,
        reverse_y=True,
    ):
        self.w = w
        self.h = h
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.reverse_y = reverse_y

        self.step_x = (x_max - x_min) / w
        self.step_y = (y_max - y_min) / h
        self.x_borders = np.linspace(x_min, x_max, w + 1)
        self.y_borders = np.linspace(y_min, y_max, h + 1)

    def get_ix(self, x, y):
        """Returns index of cell"""
        x_ix = (x - self.x_min) / self.step_x
        x_ix = int(x_ix)
        y_ix = (y - self.y_min) / self.step_y
        # print('y', np.round(self.h - y_ix, 2))
        y_ix = int(y_ix)
        if self.reverse_y:
            y_ix = self.h - y_ix - 1
        return x_ix, y_ix

    def get_coords(self, x_ix, y_ix):
        """Returns center of cell"""
        x = self.x_min + x_ix * self.step_x
        y = self.y_min + y_ix * self.step_y
        if self.reverse_y:
            y = self.y_max - y_ix * self.step_y
        return x, y

    def get_ch2grid(self, chs):
        """Returns a dict with channel names as keys and (x_ix, y_ix) as values"""
        self.ch2grid = {}
        for ch in chs:
            x, y, _ = ch["loc"][:3]
            x_ix, y_ix = self.get_ix(x, y)
            ch_name = ch["ch_name"]
            self.ch2grid[ch_name] = (x_ix, y_ix)
        return self.ch2grid

    def plot(self, **kwargs):
        for x in self.x_borders:
            for y in self.y_borders:
                plt.plot([x, x], [self.y_borders[0], self.y_borders[-1]], **kwargs)
                plt.plot([self.x_borders[0], self.x_borders[-1]], [y, y], **kwargs)

        # plot border values
        for y in self.y_borders:
            plt.text(
                self.x_borders[0],
                y,
                f"{y:.3f}",
                fontsize=8,
                ha="right",
                va="center",
                color="black",
            )

    def eval_fit(self, chs):
        """Evaluates fit of channels to grid"""
        ch2grid = self.get_ch2grid(chs)
        cell_chs = np.zeros((self.w, self.h), dtype=int)
        for x_ix in range(self.w):
            for y_ix in range(self.h):
                cell_chs[x_ix, y_ix] = len(
                    [
                        ch_name
                        for ch_name, (x, y) in ch2grid.items()
                        if (x, y) == (x_ix, y_ix)
                    ]
                )
        empty_rows = np.where(np.sum(cell_chs, axis=0) == 0)[0]
        empty_cols = np.where(np.sum(cell_chs, axis=1) == 0)[0]
        print(f"Empty rows: {len(empty_rows)}/{self.h}: {empty_rows}")
        print(f"Empty cols: {len(empty_cols)}/{self.w}: {empty_cols}")
        for ch_count in sorted(np.unique(cell_chs)):
            print(f"Cells with {ch_count} channels: {np.sum(cell_chs == ch_count)}")


def plot_head(epochs):
    afig = epochs.plot_sensors(show_names=False, show=False)
    afig.axes[0].collections[0].set_facecolor("none")
    afig.axes[0].collections[0].set_edgecolor("none")
    return afig


plot_head(bd.epochs_dict[subject])

x_coords = [ch["loc"][0] for ch in chs]
y_coords = [ch["loc"][1] for ch in chs]

# channels
plt.scatter(
    x_coords,
    y_coords,
    s=30,
    marker="o",
    facecolors="mediumvioletred",
    edgecolors="black",
    linewidths=2,
)

# channel names
for ch in chs:
    x, y, _ = ch["loc"][:3]
    plt.text(
        x,
        y + 0.005,
        ch["ch_name"].split()[0],
        fontsize=8,
        ha="center",
        va="center",
        color="black",
    )
    # plt.text(x, y-.005, f"{x_ix},{y_ix}", fontsize=8, ha='center', va='center', color='black')

grid_style = {
    "color": "black",
    "linestyle": "--",
    "linewidth": 0.5,
    "alpha": 0.5,
}

grid = BrainGrid()

plt.title(f"{dataset_name.upper()}", y=0.8)
grid.plot(**grid_style)

grid.eval_fit(chs)
