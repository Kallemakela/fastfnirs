# %%
#!%load_ext autoreload
#!%autoreload 2

from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from fastfnirs.classification.classification import get_cv_from_str
from fastfnirs.deep_learning.EEGNet import EEGNet
import logging
import mne_nirs
from fastfnirs.dataset.BrainDataset import BrainDataset
from fastfnirs.bids_to_mne import bids_to_mne
from fastfnirs.deep_learning.skorch import NestedCVSkorch
from fastfnirs.utils import get_subjects

logger = logging.getLogger(__name__)
# %% Load data

datapath = mne_nirs.datasets.fnirs_motor_group.data_path()
root_path = datapath
# %%
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
)

bd = BrainDataset(epochs_dict["tapping"]).load_epoch_data()

event_mapping = {
    "Tapping/Left": 0,
    "Tapping/Right": 1,
    # 'Control': 2,
}
bd.event_name_mapping_task = event_mapping
bd.get_full_dataset()
bd.apply_ch_selection(ch_selection="hbo")
bd.keep_classes(list(event_mapping.values()))
bd.downsample(60)
bd.to_grid(h=3, w=5, c=1)
# X_all = bd.Xc.astype(np.float32)
# y_all = bd.yc.astype(np.int64)
X_all = bd.Xc.astype(np.float32)
y_all = bd.yc.astype(np.int64)
print(X_all.shape, np.unique(y_all, return_counts=True))
# %% Classification

net_params = {
    "n_chs": bd.Xc.shape[1],
    "num_timesteps": bd.Xc.shape[2],
    "num_classes": len(event_mapping),
    "kernel_size_1": 12,
    "stride_1": 2,
}
base_model = EEGNet
net_params = {f"module__{k}": v for k, v in net_params.items()}

outer_cv_str = "gk5"  # group k-fold
outer_cv = get_cv_from_str(outer_cv_str)
outer_cv = list(outer_cv.split(X_all, y_all, bd.groups))
inner_cv = None  # no hyperparam tuning

# croNestedCVSkorchor all epochs
cv = NestedCVSkorch(
    outer_cv=outer_cv,
    inner_cv=inner_cv,
    net_params=net_params,
    base_model=base_model,
    max_epochs=200,
    scoring=accuracy_score,
    get_model_kwargs={
        "criterion": nn.CrossEntropyLoss,
        "batch_size": 32,
    },
)
cv.fit(X_all, y_all)
# %%

plt.plot(cv.test_scores.mean(axis=0), label=f"mean")
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Accuracy")

# %%
