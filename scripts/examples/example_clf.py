# %%
#!%load_ext autoreload
#!%autoreload 2

import numpy as np
import logging
import mne_nirs

from fastfnirs.dataset.BrainDataset import BrainDataset
from fastfnirs.bids_to_mne import bids_to_mne
from fastfnirs.classification.helpers import epoch_classification
from fastfnirs.visualization import plot_evoked
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
# %%
bd = BrainDataset(epochs_dict["tapping"]).load_epoch_data()
# %% Classification

event_mapping = {
    "Tapping/Left": 0,
    "Tapping/Right": 1,
    # 'Control': 2,
}
bd.event_name_mapping_task = event_mapping
repeats_ind = 2
repeats_cross = 2
clf_res = epoch_classification(
    bd,
    features=["MV"],
    n_windows=3,
    print_report=True,
    ind_cv=f"looeco_r{repeats_ind}",
    cross_cv=f"gk3_r{repeats_cross}",
)
# %% Compare evoked responses for classes

plot_evoked(epochs_dict["tapping"], conditions=list(event_mapping.keys()))
# %% Connect to metadata

md = bd.get_metadata()
keep_cols = ["subject", "trial_type"]
md = md[keep_cols]
md = md[md["trial_type"].isin(list(event_mapping.keys()))]
md["target"] = md["trial_type"].map(event_mapping)
md["cross_preds"] = list(clf_res["cross_preds_arr"])
md["ind_preds"] = list(clf_res["ind_preds_arr"])


print(f"{'Subject':<10} {'Cross':<10} {'Withn':<10}")
for subject in md["subject"].unique():
    subject_md = md[md["subject"] == subject]
    sub_cross_preds = np.stack(subject_md["cross_preds"].values)  # (n_trials, repeats)
    target_repeated = np.repeat(
        subject_md["target"].values[:, None], repeats_cross, axis=1
    )  # (n_trials, repeats)
    cross_acc = (sub_cross_preds == target_repeated).mean()

    sub_ind_preds = np.stack(subject_md["ind_preds"].values)  # (n_trials, repeats)
    target_repeated = np.repeat(
        subject_md["target"].values[:, None], repeats_ind, axis=1
    )  # (n_trials, repeats)
    ind_acc = (sub_ind_preds == target_repeated).mean()
    print(f"{subject:<10} {cross_acc:<10.2f} {ind_acc:<10.2f}")


# %%
