import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import LeaveOneGroupOut
from fastfnirs.utils import reverse_dict, combine_event_map
from fastfnirs.dataset.BrainDataset import BrainDataset
from fastfnirs.classification.classification import (
    extract_features_from_raw,
    concatenate_features,
    get_model,
    get_cv_from_str,
    cross_val_predict_repeated,
)


def print_results(y, preds, event2name=None):
    yc = np.concatenate([yi for yi in y.values()])

    # If preds is a 2D array with the same number of rows as yc, and is integer type, then its repeated predictions
    if len(preds.shape) == len(yc.shape) + 1 and np.issubdtype(preds.dtype, np.integer):
        yc = np.repeat(yc, preds.shape[1])
        preds = preds.reshape(-1)

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
    bd: BrainDataset,
    features=["MV"],
    n_windows=1,
    ch_selection="hbo",
    print_report=True,
    seed=1,
    **kwargs,
):
    """
    Performs subject-specific and cross-subject classification of epochs.
    Uses `bd.event_name_mapping_task` to map event names to classes.

    Parameters
    ----------
    epochs_dict : dict
            Dictionary of epochs, with subject IDs as keys.
    features : list, optional
            List of features to extract from epochs. The default is ['MV'].
    n_windows : int, optional
            Number of windows to split each epoch into. The default is 1.
    ch_selection : str, optional
            Channel selection. The default is 'hbo'.
    """
    predict_cross = len(bd.epochs_dict) > 1

    if "verbose" in kwargs:
        bd.verbose = int(kwargs["verbose"])
    bd.get_full_dataset()
    bd.keep_classes(classes=list(bd.event_name_mapping_task.values()))
    bd.filter_by_class_count()
    bd.apply_ch_selection(ch_selection=ch_selection)
    Xr, y = bd.X, bd.y

    X = extract_features_from_raw(Xr, features=features, n_windows=n_windows)
    X = concatenate_features(X)
    Xc = np.concatenate([*X.values()])
    yc = np.concatenate([*y.values()])
    subject_ids = np.concatenate(
        [np.full(len(yi), subject) for subject, yi in y.items()]
    )
    if "model" in kwargs:
        model = kwargs["model"]
    else:
        model = get_model(n_classes=len(np.unique(yc)), seed=seed)

    ind_preds = []
    for subject in X.keys():
        Xs, ys, subject_ids = X[subject], y[subject], subject_ids
        if "ind_cv" in kwargs:
            sub_cv = kwargs["ind_cv"]
            if isinstance(sub_cv, str):
                sub_cv = get_cv_from_str(sub_cv, y=ys, seed=seed)
        else:
            sub_cv = get_cv_from_str("sk5_r1", y=ys, seed=seed)
        sub_cv = list(sub_cv.split(Xs, ys))
        preds = cross_val_predict_repeated(model, Xs, ys, splits=sub_cv)
        ind_preds.append((subject, preds, ys))

    ind_preds_arr = np.concatenate([o[1] for o in ind_preds])

    if "cross_cv" in kwargs:
        cross_cv = get_cv_from_str(kwargs["cross_cv"], y=yc, seed=seed)
    else:
        cross_cv = LeaveOneGroupOut()

    if predict_cross:
        cross_cv = list(cross_cv.split(Xc, yc, subject_ids))
        cross_preds_arr = cross_val_predict_repeated(
            model, Xc, yc, splits=cross_cv, n_jobs=-1
        )

    combined_event_map = combine_event_map(bd.event_name_mapping_task)

    if print_report:
        print(
            f"X.shape: {Xc.shape}, y label counts: {np.unique(yc, return_counts=True)}"
        )
        print(f"Model: {model}")
        print()
        print(f"Individual subject classification:")
        print_results(y, ind_preds_arr, event2name=reverse_dict(combined_event_map))
        if predict_cross:
            print(f"Cross-subject classification:")
            print_results(
                y, cross_preds_arr, event2name=reverse_dict(combined_event_map)
            )

    return {
        "ind_preds_arr": ind_preds_arr,
        "cross_preds_arr": cross_preds_arr if predict_cross else None,
        "y": yc,
        "ind_preds": ind_preds,
    }
