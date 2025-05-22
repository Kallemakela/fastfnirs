# %%
"""
Example of nested cross-validation and analysis of results.
"""
#!%load_ext autoreload
#!%autoreload 2

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import mne_nirs

from fastfnirs.dataset.feature_extraction import fNIRSFeatureExtractor
from fastfnirs.dataset import BrainDataset
from fastfnirs.bids_to_mne import bids_to_mne
from fastfnirs.utils import get_subjects
from fastfnirs.classification import get_cv_from_str
from fastfnirs.classification.nested_cv import NestedCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from tqdm import tqdm

# %%

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
epochs_dict = epochs_dict["tapping"]
# %%

bd = BrainDataset(epochs_dict).load_epoch_data()

event_mapping = {
    "Tapping/Left": 0,
    "Tapping/Right": 1,
    # "Control": 2,
}
bd.event_name_mapping_task = event_mapping
bd.get_full_dataset().keep_classes(list(event_mapping.values())).apply_ch_selection(
    ch_selection="hbo"
)  # .extract_features(features=["MV"], n_windows=3)

# %%


model = Pipeline(
    [
        ("fe", fNIRSFeatureExtractor()),
        ("scaler", StandardScaler()),
        # ("clf", LogisticRegression(max_iter=10000, random_state=42)),
        # ("clf", RandomForestClassifier(random_state=42)),
        # ("clf", SVC(random_state=42)),
        ("clf", LinearDiscriminantAnalysis(shrinkage="auto", solver="lsqr")),
    ]
)
# param_grid = {"clf__C": [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 10000]}
# param_grid = {
#     "clf__n_estimators": [10, 50, 100],
#     "clf__max_depth": [None, 30, 50],
# }
# param_grid = {
#     "clf__C": [0.0001, 0.01, 1, 100, 10000],
#     "clf__kernel": ["linear", "rbf"],
# }

param_grid = {
    "fe__features": ["MV", "STD", "SLO", ["MV", "SLO"]],
    "fe__n_windows": [1, 2, 4, 10, 20],
}
outer_cv_str = "sk5_r1"
inner_cv_str = "sk5_r1"


sub_scores = []
sub_split_info = []
for subject in list(bd.X.keys()):
    Xs, ys = bd.X[subject], bd.y[subject]
    outer_cv = get_cv_from_str(outer_cv_str, seed=42)
    inner_cv = get_cv_from_str(inner_cv_str, seed=42)
    cv = NestedCV(model, param_grid, outer_cv, inner_cv)
    cv.fit(Xs, ys)
    sub_scores.append(cv.test_scores)
    sub_split_info.append(cv.split_info)

print(f"Score: {np.mean(sub_scores):.3f} ± {np.std(sub_scores):.3f}")
# %%


def check_param_order(split_info, correct_order=None):
    for si in split_info:
        if correct_order is None:
            correct_order = si["params"]
        else:
            assert si["params"] == correct_order, "Parameters don't match"
    return correct_order


def pretty_param_str(params):
    """Returns a short string representation of the parameters"""
    pretty_str = ""
    for k, v in params.items():
        part, val_name = k.split("__")
        val_name_first_letters = "".join([v[0] for v in val_name.split("_")])
        if v is None:
            v = "no"
        pretty_str += f"{val_name_first_letters}={v} "
    return pretty_str


correct_order = None
for split_info in sub_split_info:
    correct_order = check_param_order(split_info, correct_order)

val_scores = []
param_scores = []
param_counter = np.zeros(len(correct_order))
for split_info in sub_split_info:
    for si in split_info:
        split_scores = si["mean_test_score"]
        val_scores.append(split_scores)
        highest_score_ix = np.argmax(split_scores)
        param_counter[highest_score_ix] += 1

val_scores = np.array(val_scores)
for pi, params in enumerate(correct_order):
    param_scores.append(
        {
            "params": pretty_param_str(params),
            "val_score": val_scores[:, pi],
            "count": param_counter[pi],
        }
    )
param_scores = pd.DataFrame(param_scores)
param_scores = param_scores.astype({"params": "str"})
param_scores = param_scores.explode("val_score")
# %%

sns.boxplot(y="val_score", x="params", data=param_scores)
plt.xticks(rotation=90)
plt.tight_layout()

print(param_scores.groupby("params").mean().sort_values("val_score", ascending=False))
# %%
