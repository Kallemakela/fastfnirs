# %%
import numpy as np
import matplotlib.pyplot as plt

from fastfnirs.dataset.BrainDataset import BrainDataset
from fastfnirs.utils import get_all_subjects

# %%
dataset_name = "nemo"
subjects = get_all_subjects(dataset_name)
subject = subjects[0]
bd = (
    BrainDataset.load_bd(dataset_name, subjects=subjects)
    .get_full_dataset()
    .downsample()
    .apply_ch_selection()
    .to_grid()
)

if dataset_name == "nemo":
    epoch_ix = 5
    vmin = -0.35
elif dataset_name == "mima":
    epoch_ix = 1
    vmin = -0.55
elif dataset_name == "bnci":
    epoch_ix = 8
    vmin = -2e5

times = np.array([1, 4, 7, 10])
times_ix = np.round((60 / 12) * times).astype(int)

vmax = -vmin

epoch = bd.epochs_dict[subject][epoch_ix]
gd = bd.X[subject][epoch_ix, 0] * 1e6

afig = epoch.average("hbo").plot_topomap(
    times=times,
    time_unit="s",
    extrapolate="local",
    vlim=(vmin, vmax),
    time_format="%ds",
    contours=0,
)
plt.tight_layout()
# afig.savefig('topo_a.png', dpi=300)
plt.show()

plt.figure(figsize=(12, 3))
for ii, time_ix in enumerate(times_ix):
    g2d = gd[::-1, :, time_ix]
    plt.subplot(1, len(times_ix), ii + 1)
    # plt.title(f'time: {times[ii]}s')
    plt.imshow(g2d, cmap="RdBu_r", vmin=vmin, vmax=vmax)
    plt.xticks([])
    plt.yticks([])

plt.tight_layout()
# plt.savefig('topo_b.png', dpi=300)
plt.show()
# %%
gd
