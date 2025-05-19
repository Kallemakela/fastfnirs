from collections import defaultdict
import numpy as np
from sklearn.base import clone
from sklearn.model_selection import cross_val_predict
from fastfnirs.utils.basic import split_list


def detect_repeats(splits):
    """Detects number of repeats in splits"""
    counter = defaultdict(int)
    for _, test_index in splits:
        for i in test_index:
            counter[i] += 1

    count = np.array(list(counter.values()))
    assert np.all(count == count[0]), "All repeats must be the same"
    return count[0]


def _validate_repeated_splits(splits_by_repeat):
    n_repeats = len(splits_by_repeat)
    all_ixs = set(
        [
            i
            for repeat_splits in splits_by_repeat
            for _, test_index in repeat_splits
            for i in test_index
        ]
    )
    for repeat_splits in splits_by_repeat:
        counter = defaultdict(int)
        for _, test_index in repeat_splits:
            for i in test_index:
                counter[i] += 1

        for ix in all_ixs:
            assert counter[ix] == 1, "All ixs should appear once in each repeat"


def cross_val_predict_repeated(model, X, y, splits, **kwargs):
    """
    Wrapper for cross_val_predict that allows for repeated cross-validation.

    Returns an array of shape (n_samples, n_repeats, ...). The predictions are ordered, so you can pair them e.g. by np.c_[meta_y, preds].
    """
    n_repeats = detect_repeats(splits)
    splits_by_repeat = split_list(splits, n_repeats)
    _validate_repeated_splits(splits_by_repeat)
    preds = {}
    for i, repeat_splits in enumerate(splits_by_repeat):
        preds[i] = cross_val_predict(clone(model), X, y, cv=repeat_splits, **kwargs)
    preds = np.array(list(preds.values()))
    return preds.transpose(1, 0, *preds.shape[2:])
