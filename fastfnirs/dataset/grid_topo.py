# %%
import numpy as np
import matplotlib.pyplot as plt
import mne_nirs

from fastfnirs.dataset.BrainDataset import BrainDataset
from fastfnirs.bids_to_mne import bids_to_mne
from fastfnirs.utils import get_subjects

# %%
# dataset_name = "nemo"
# subjects = get_subjects(dataset_name)
dataset_name = "motor"
root_path = mne_nirs.datasets.fnirs_motor_group.data_path()
subjects = get_subjects(root_path)
process_raw_kwargs = dict(
    h_freq=0.1,
    l_freq=0.01,
    sci_threshold=0.5,
    verbose=False,
)
epochs_kwargs = dict(
    tmin=-5,
    tmax=12,
)
epochs_dict = bids_to_mne(
    root_path,
    process_raw_kwargs=process_raw_kwargs,
    epochs_kwargs=epochs_kwargs,
)["tapping"]
# %%
bd = (
    BrainDataset(epochs_dict)
    .get_full_dataset()
    .downsample()
    .apply_ch_selection()
    .to_grid(h=5, w=15, c=1)
)
# %%

if dataset_name == "nemo":
    epoch_ix = 5
    vmin = -0.35
elif dataset_name == "mima":
    epoch_ix = 1
    vmin = -0.55
elif dataset_name == "bnci":
    epoch_ix = 8
    vmin = -2e5
elif dataset_name == "motor":
    epoch_ix = 0
    vmin = -0.65

times = np.array([1, 4, 7, 10])
times_ix = np.round((60 / 12) * times).astype(int)

vmax = -vmin

subject = subjects[0]
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
