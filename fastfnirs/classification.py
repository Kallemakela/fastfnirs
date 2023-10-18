import numpy as np
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
import logging

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    cross_val_predict,
    RepeatedStratifiedKFold,
    LeaveOneGroupOut,
)
from sklearn.metrics import confusion_matrix, classification_report
from fastfnirs.data import BrainDataset

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


def reverse_dict(d):
    return {v: k for k, v in d.items()}


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


def extract_features_from_raw(X, features=["MV"], n_windows=3):
    Xf = defaultdict(list)
    for subject, d in X.items():
        sXf = []
        n_epochs, n_samples, n_channels = d.shape
        L = n_samples // n_windows
        for wi in range(n_windows):
            wd = d[:, wi * L : (wi + 1) * L, :]
            if "IAV" in features:
                sXf.append(np.sum(np.abs(wd), axis=1))
            if "MAV" in features:
                sXf.append(np.mean(np.abs(wd), axis=1))
            if "MV" in features:
                sXf.append(np.mean(wd, axis=1))
            if "PMN" in features:
                mu = np.mean(wd, axis=1)
                centered_wd = wd - mu[:, None, :]
                PMN = 0
                for si in range(wd.shape[1] - 1):
                    PMN += centered_wd[:, si, :] * centered_wd[:, si + 1, :] < 0
                sXf.append(PMN)
            if "PZN" in features:
                PZN = 0
                for si in range(wd.shape[1] - 1):
                    PZN += wd[:, si, :] * wd[:, si + 1, :] < 0
                sXf.append(PZN)
            if "STD" in features:
                sXf.append(np.std(wd, axis=1))
            if "polyfit_coef_1" in features:
                perm_wd = np.swapaxes(wd, 0, 1).reshape(wd.shape[1], -1)
                pf = np.polyfit(np.arange(perm_wd.shape[0]), perm_wd, 1)[0]
                sXf.append(pf.reshape(n_epochs, -1))
        sXf = np.array(sXf).transpose(1, 2, 0).reshape(n_epochs, -1)
        Xf[subject] = sXf
    return Xf


def filter_classes(Xr, y, include_classes):
    for subject in Xr.keys():
        include_ix = np.isin(y[subject], include_classes)
        Xr[subject] = Xr[subject][include_ix]
        y[subject] = y[subject][include_ix]
    return Xr, y


def get_cv(y, seed=1, **kwargs):
    _, label_counts = np.unique(y, return_counts=True)
    cv = RepeatedStratifiedKFold(
        n_splits=np.min(label_counts), n_repeats=1, random_state=seed, **kwargs
    )
    return cv


def ind_clf(X, y, model=None):
    if model is None:
        yc = np.concatenate([yi for yi in y.values()])
        model = get_model(n_classes=len(np.unique(yc)))
    output = []
    for subject in X.keys():
        preds = cross_val_predict(
            model, X[subject], y[subject], n_jobs=-1, cv=get_cv(y[subject])
        )
        output.append((subject, preds, y[subject]))  # , epoch_ids[subject]))
    ind_preds = np.concatenate([o[1] for o in output])
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


def print_results(y, preds, event2name=None):
    yc = np.concatenate([yi for yi in y.values()])
    print(
        classification_report(
            yc, preds, target_names=[event2name[c] for c in np.unique(yc)]
        )
    )
    print(confusion_matrix(yc, preds))
    print()
    for condition in np.unique(yc):
        print(
            f"{event2name[condition]:5}: {np.mean(preds[yc == condition] == condition):.3f}"
        )


def epoch_classification(
    epochs_dict,
    event_mapping,
    features=["MV"],
    n_windows=1,
    ch_selection="hbo",
    print_report=True,
    seed=1,
    **kwargs,
):
    """
    Performs subject-specific and cross-subject classification of epochs.

    Parameters
    ----------
    epochs_dict : dict
            Dictionary of epochs, with subject IDs as keys.
    event_mapping : dict
            Dictionary mapping event names to integers.
    features : list, optional
            List of features to extract from epochs. The default is ['MV'].
    n_windows : int, optional
            Number of windows to split each epoch into. The default is 1.
    ch_selection : str, optional
            Channel selection. The default is 'hbo'.
    """
    predict_cross = len(epochs_dict) > 1

    bd = BrainDataset(epochs_dict)
    bd.event_name_mapping_task = event_mapping
    bd.get_full_dataset()
    bd.filter_by_class_count()
    bd.apply_ch_selection(ch_selection=ch_selection)
    Xr, y = bd.X, bd.y
    X = extract_features_from_raw(Xr, features=features, n_windows=n_windows)

    Xc = np.concatenate([*X.values()])
    yc = np.concatenate([*y.values()])
    subject_ids = np.concatenate(
        [np.full(len(yi), subject) for subject, yi in y.items()]
    )
    if "model" in kwargs:
        model = kwargs["model"]
    else:
        model = get_model(n_classes=len(np.unique(yc)), seed=seed)

    output = []
    for subject in X.keys():
        if "ind_cv" in kwargs:
            sub_cv = kwargs["ind_cv"]
        else:
            sub_cv = get_cv(y[subject], seed=seed)

        preds = cross_val_predict(model, X[subject], y[subject], n_jobs=-1, cv=sub_cv)
        output.append((subject, preds, y[subject]))
    ind_preds = np.concatenate([o[1] for o in output])

    if "cross_cv" in kwargs:
        cross_cv = kwargs["cross_cv"]
    else:
        cross_cv = LeaveOneGroupOut()

    if predict_cross:
        cross_preds = cross_val_predict(
            model, Xc, yc, n_jobs=-1, cv=cross_cv.split(Xc, yc, subject_ids)
        )

    if print_report:
        print(
            f"X.shape: {Xc.shape}, y label counts: {np.unique(yc, return_counts=True)}"
        )
        print(f"Model: {model}")
        print()
        print(f"Individual subject classification:")
        print_results(y, ind_preds, event2name=reverse_dict(event_mapping))
        if predict_cross:
            print(f"Cross-subject classification:")
            print_results(y, cross_preds, event2name=reverse_dict(event_mapping))

    return {
        "ind_preds": ind_preds,
        "cross_preds": cross_preds if predict_cross else None,
        "y": yc,
    }
