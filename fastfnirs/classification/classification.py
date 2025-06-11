import re
import numpy as np
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
import logging
from sklearn.model_selection import BaseCrossValidator
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    cross_val_predict,
    KFold,
    RepeatedStratifiedKFold,
    LeaveOneGroupOut,
    LeaveOneOut,
    RepeatedKFold,
)
from customCV.group import (
    RepeatedUniqueFoldGroupKFoldPG as RepeatedUniqueFoldGroupKFold,
    GroupCVWrapper,
)
from sklearn.metrics import confusion_matrix, classification_report
from fastfnirs.classification.sklearn_helpers import cross_val_predict_repeated
from fastfnirs.dataset.BrainDataset import BrainDataset
from fastfnirs.utils import combine_event_map


logger = logging.getLogger(__name__)


def get_epochs_dfs(epochs_dict, disable_tqdm=True):
    """
    Combines epoch data and metadata from all subjects to dataframes `epochs_df` and `epochs_metadata_df`.

    Parameters
    ----------
    epochs_dict : dict
            Dictionary subject -> epochs.

    Returns
    -------
    epochs_df : pandas.DataFrame
            Dataframe containing all epoch data.
    epochs_metadata_df : pandas.DataFrame
            Dataframe containing all epoch metadata.
    """
    n_epochs = 0
    subject_edfs = []
    epoch_metadata_dfs = []
    for subject, subject_epochs in tqdm(
        list(epochs_dict.items()), disable=disable_tqdm
    ):
        subject_edf = subject_epochs.to_data_frame(verbose=False)
        subject_metadata = subject_epochs.metadata.copy()
        if "epoch" not in subject_metadata.columns:
            subject_metadata["epoch"] = subject_metadata.index
        missing_epochs = set(subject_metadata["epoch"].unique()) - set(
            subject_edf["epoch"].unique()
        )
        if len(missing_epochs) > 0:
            logger.warn(
                f"to_data_frame() dropped epochs {missing_epochs} for subject {subject}. Correcting epoch ids."
            )
            correction = [
                sum([x > epoch for epoch in missing_epochs])
                for x in subject_edf["epoch"]
            ]
            subject_edf["epoch"] -= correction
        subject_edf["epoch"] += n_epochs  # new epoch ids
        subject_metadata["epoch"] += n_epochs
        subject_edf = pd.merge(subject_edf, subject_metadata, how="left", on="epoch")
        if subject_edf["subject"].isna().any():
            subject = subject if subject.startswith("sub-") else f"sub-{subject}"
            subject_edf["subject"] = subject
            logger.warn(
                f"Found nan subject for subject {subject}. Setting to {subject}."
            )
        subject_edf = subject_edf[
            [
                "time",
                "subject",
                "epoch",
                "condition",
                "bad_channels",
                "is_bad_epoch",
                "bad_epoch_reason",
            ]
            + subject_epochs.ch_names
        ]
        subject_edfs.append(subject_edf)
        epoch_metadata_dfs.append(subject_metadata)
        n_epochs += len(subject_epochs)
    epochs_df = pd.concat(subject_edfs, ignore_index=True)
    epochs_metadata_df = pd.concat(epoch_metadata_dfs, ignore_index=True)
    return epochs_df, epochs_metadata_df


def get_model(model_name="lda", n_classes=None, seed=1):
    if model_name == "lda":
        clf = LinearDiscriminantAnalysis(
            solver="lsqr",
            shrinkage="auto",
            # priors=np.array([1/n_classes]*n_classes)
        )
    elif model_name == "lr" or model_name == "logreg":
        clf = LogisticRegression(
            penalty="l2",
            # C=0.25,
            C=0.4,
            max_iter=10000,
            n_jobs=-1,
            random_state=seed,
        )

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", clf),
        ]
    )
    return model


def get_channels_by_selection(channels, ch_selection):
    if isinstance(ch_selection, str):
        channels = [ch for ch in channels if ch_selection in ch]
    return channels


def get_feature_names(channels, features=["MV"], n_windows=3, ch_selection="hbo"):
    """
    Returns feature names in the same order as `extract_features_from_raw`.
    """
    channels = get_channels_by_selection(channels, ch_selection)
    return np.array(
        [
            f"{ch} {f}_{wi:03d}"
            for ch in channels
            for wi in range(n_windows)
            for f in sorted(features)
        ]
    )


def get_raw_dataset_from_df(edf, channels, event_mapping):
    Xr = defaultdict(list)
    y = defaultdict(list)
    epoch_ids = defaultdict(list)
    included_epochs = edf["epoch"].unique()
    for epoch_id in sorted(included_epochs):
        epoch_df = edf[(edf["epoch"] == epoch_id) & (edf["time"] > 0)]
        subject = epoch_df["subject"].iloc[0]
        Xr[subject].append(epoch_df[channels].values)
        y[subject].append(epoch_df["condition"].map(event_mapping).iloc[0])
        epoch_ids[subject].append(epoch_id)
    for subject in Xr.keys():
        Xr[subject] = np.array(Xr[subject])
        y[subject] = np.array(y[subject])
        epoch_ids[subject] = np.array(epoch_ids[subject])
    return Xr, y, epoch_ids


def extract_features_from_array(d, features=["MV"], n_windows=3):
    """
    d : np.ndarray of shape (n_epochs, n_channels, n_samples)
    """
    sXf = defaultdict(list)
    n_epochs, n_channels, n_samples = d.shape
    L = n_samples // n_windows

    for wi in range(n_windows):
        wd = d[..., wi * L : (wi + 1) * L]
        if "IAV" in features:
            sXf["IAV"].append(np.sum(np.abs(wd), axis=-1))
        if "MAV" in features:
            sXf["MAV"].append(np.mean(np.abs(wd), axis=-1))
        if "MV" in features:
            sXf["MV"].append(np.mean(wd, axis=-1))
        if "PMN" in features:
            mu = np.mean(wd, axis=-1)
            centered_wd = wd - mu[..., None]
            PMN = 0
            for si in range(wd.shape[-1] - 1):
                PMN += centered_wd[..., si] * centered_wd[..., si + 1] < 0
            sXf["PMN"].append(PMN)
        if "PZN" in features:
            PZN = 0
            for si in range(wd.shape[-1] - 1):
                PZN += wd[..., si] * wd[..., si + 1] < 0
            sXf["PZN"].append(PZN)
        if "STD" in features:
            sXf["STD"].append(np.std(wd, axis=-1))
        if "polyfit_coef_1" in features:
            perm_wd = wd.transpose(2, 0, 1).reshape(wd.shape[-1], -1)
            poly_x = np.arange(perm_wd.shape[0])
            pf = np.polyfit(poly_x, perm_wd, 1)[0]
            pf = pf.reshape(n_epochs, -1)
            sXf["polyfit_coef_1"].append(pf)
        if "AMP" in features:
            sXf["AMP"].append(np.max(wd, axis=-1) - np.min(wd, axis=-1))

    return sXf


def extract_features_from_raw(X, features=["MV"], n_windows=3):
    """
    X : dict
        Dictionary subject -> x. x shape: (n_epochs, n_channels, n_samples)
    """
    Xf = {}
    for subject, d in X.items():
        Xf[subject] = extract_features_from_array(
            d, features=features, n_windows=n_windows
        )
    return Xf


def concatenate_features(Xf):
    for subject in Xf.keys():
        xfc = np.array(
            list(Xf[subject].values())
        )  # (n_features, n_windows, n_epochs, n_channels)
        xfc = xfc.transpose(2, 3, 1, 0)  # (n_epochs, n_channels, n_windows, n_features)
        xfc = xfc.reshape(xfc.shape[0], -1)
        Xf[subject] = xfc
    return Xf


def extract_features_simple(*args, **kwargs):
    """
    Wrapper for `extract_features_from_array` that only returns the features.
    Returns them in (epochs, channels, features) shape.
    """
    Xf = extract_features_from_array(*args, **kwargs)
    Xf = np.array([f for ft in Xf.values() for f in ft])
    Xf = Xf.transpose(1, 2, 0)  # (n_epochs, n_channels, n_features)
    return Xf


def filter_classes(Xr, y, include_classes):
    for subject in Xr.keys():
        include_ix = np.isin(y[subject], include_classes)
        Xr[subject] = Xr[subject][include_ix]
        y[subject] = y[subject][include_ix]
    return Xr, y


def get_cv_from_str(cv_str, n=None, y=None, seed=None, X=None, groups=None, **kwargs):
    if re.match(r"k\d+", cv_str):
        k = int(cv_str[1:])
        if seed is None:
            return KFold(n_splits=k)
        else:
            return KFold(n_splits=k, shuffle=True, random_state=seed)
    elif cv_str == "loo":
        return KFold(n_splits=n)
    # "looeco_r2"
    elif cv_str.startswith("looeco"):
        parts = cv_str.split("_")
        n_repeats = int(parts[1][1:]) if len(parts) == 2 else 1
        _, label_counts = np.unique(y, return_counts=True)
        return RepeatedStratifiedKFold(
            n_splits=np.min(label_counts),
            n_repeats=n_repeats,
            random_state=seed,
            **kwargs,
        )
    # sk10
    elif cv_str.startswith("sk"):
        parts = cv_str.split("_")
        n_splits = int(parts[0][2:])
        n_repeats = int(parts[1][1:]) if len(parts) == 2 else 1
        return RepeatedStratifiedKFold(
            n_splits=n_splits, n_repeats=n_repeats, random_state=seed, **kwargs
        )
    # loso
    elif cv_str.startswith("loso"):
        base_cv = LeaveOneOut()
        return GroupCVWrapper(base_cv)
    # gk2_r2
    elif cv_str.startswith("gk"):
        parts = cv_str.split("_")
        n_splits = int(parts[0][2:])
        n_repeats = int(parts[1][1:]) if len(parts) == 2 else 1
        base_cv = RepeatedKFold(
            n_splits=n_splits, n_repeats=n_repeats, random_state=seed, **kwargs
        )
        return GroupCVWrapper(base_cv)

    # ugk2_r2
    elif cv_str.startswith("ugk"):
        parts = cv_str.split("_")
        n_splits = int(parts[0][3:])
        n_repeats = int(parts[1][1:]) if len(parts) == 2 else 1
        return RepeatedUniqueFoldGroupKFold(
            n_splits=n_splits, n_repeats=n_repeats, random_state=seed, **kwargs
        )
    else:
        raise ValueError(f"Unknown cv_str: {cv_str}")


def get_cv_splits_from_arg(cv_arg, **kwargs):
    get_cv_args = {
        "y": kwargs.get("y"),
        "seed": kwargs.pop("seed", None),
    }
    if isinstance(cv_arg, str):
        return list(get_cv_from_str(cv_arg, **get_cv_args).split(**kwargs))
    elif isinstance(cv_arg, list):
        return cv_arg
    elif isinstance(cv_arg, BaseCrossValidator):
        return list(cv_arg.split(**kwargs))
    else:
        raise ValueError(f"Unknown cv_arg: {cv_arg}")


def ind_clf(X, y, model=None, cv_str="looeco_r1"):
    """
    Performs within-subject classification. Returns a list of tuples (subject, preds, y).
    """
    if model is None:
        yc = np.concatenate([yi for yi in y.values()])
        model = get_model(n_classes=len(np.unique(yc)))
    ind_preds = []
    for subject in X.keys():
        Xs, ys = X[subject], y[subject]
        splits = list(get_cv_from_str(cv_str, y=ys).split(Xs, ys))
        preds = cross_val_predict_repeated(
            model,
            Xs,
            ys,
            n_jobs=-1,
            splits=splits,
        )
        ind_preds.append((subject, preds, ys))
    return ind_preds


def cross_clf(X, y, model=None):
    Xc = np.concatenate([*X.values()])
    yc = np.concatenate([*y.values()])
    if model is None:
        model = get_model(n_classes=len(np.unique(yc)))
    subject_ids = np.concatenate(
        [np.full(len(yi), subject) for subject, yi in y.items()]
    )
    cross_preds = cross_val_predict(
        model, Xc, yc, n_jobs=-1, cv=LeaveOneGroupOut().split(Xc, yc, subject_ids)
    )
    return cross_preds
