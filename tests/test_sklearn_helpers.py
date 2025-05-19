import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from fastfnirs.classification.sklearn_helpers import cross_val_predict_repeated


def test_basic_functionality():
    # Create simple dataset
    X = np.array([[1], [2], [3], [4]])
    y = np.array([0, 0, 1, 1])

    # Create splits with 2 repeats, ensuring each training fold has both classes
    splits = [
        # Repeat 1
        (
            np.array([0, 2]),
            np.array([1, 3]),
        ),  # Fold 1: Train [0,2] (class 0,1), Test [1,3] (class 0,1)
        (
            np.array([1, 3]),
            np.array([0, 2]),
        ),  # Fold 2: Train [1,3] (class 0,1), Test [0,2] (class 0,1)
        # Repeat 2
        (
            np.array([0, 3]),
            np.array([1, 2]),
        ),  # Fold 1: Train [0,3] (class 0,1), Test [1,2] (class 0,1)
        (
            np.array([1, 2]),
            np.array([0, 3]),
        ),  # Fold 2: Train [1,2] (class 0,1), Test [0,3] (class 0,1)
    ]

    model = LogisticRegression()
    predictions = cross_val_predict_repeated(model, X, y, splits)

    # Check shape: (n_repeats, n_samples)
    assert predictions.shape == (4, 2)
    # Check that predictions are binary for logistic regression
    assert np.all(np.isin(predictions, [0, 1]))


def test_predict_proba():
    X = np.array([[1], [2], [3], [4]])
    y = np.array([0, 0, 1, 1])

    splits = [
        # Repeat 1
        (np.array([0, 2]), np.array([1, 3])),
        (np.array([1, 3]), np.array([0, 2])),
        # Repeat 2
        (np.array([0, 3]), np.array([1, 2])),
        (np.array([1, 2]), np.array([0, 3])),
    ]

    model = LogisticRegression()
    predictions = cross_val_predict_repeated(
        model, X, y, splits, method="predict_proba"
    )

    # Check shape: (n_repeats, n_samples, n_classes)
    assert predictions.shape == (4, 2, 2)
    # Check that probabilities sum to 1
    assert np.allclose(predictions.sum(axis=2), 1)


def test_invalid_splits():
    X = np.array([[1], [2], [3], [4]])
    y = np.array([0, 0, 1, 1])

    # Invalid splits: sample 0 appears twice in first repeat
    invalid_splits = [
        # Repeat 1
        (np.array([0, 2]), np.array([1, 3])),
        (np.array([0, 1]), np.array([0, 3])),  # Invalid: sample 0 appears twice
        # Repeat 2
        (np.array([0, 3]), np.array([1, 2])),
        (np.array([1, 2]), np.array([0, 3])),
    ]

    model = LogisticRegression()
    with pytest.raises(AssertionError):
        cross_val_predict_repeated(model, X, y, invalid_splits)


def test_unequal_repeats():
    X = np.array([[1], [2], [3], [4]])
    y = np.array([0, 0, 1, 1])

    # Invalid splits: sample 0 appears once in first repeat, twice in second
    invalid_splits = [
        # Repeat 1
        (np.array([0, 2]), np.array([1, 3])),
        (np.array([1, 3]), np.array([0, 2])),
        # Repeat 2
        (np.array([0, 3]), np.array([1, 2])),
        (np.array([1, 2]), np.array([0, 3])),
        (np.array([0, 1]), np.array([2, 3])),  # Extra fold in second repeat
    ]

    model = LogisticRegression()
    with pytest.raises(AssertionError):
        cross_val_predict_repeated(model, X, y, invalid_splits)


def test_different_model():
    X = np.array([[1], [2], [3], [4]])
    y = np.array([0, 0, 1, 1])

    splits = [
        # Repeat 1
        (np.array([0, 3]), np.array([1, 2])),
        (np.array([1, 2]), np.array([0, 3])),
        # Repeat 2
        (np.array([0, 2]), np.array([1, 3])),
        (np.array([1, 3]), np.array([0, 2])),
    ]

    model = RandomForestClassifier()
    predictions = cross_val_predict_repeated(model, X, y, splits)

    assert predictions.shape == (4, 2)
    # LDA predictions should be class labels
    assert np.all(np.isin(predictions, [0, 1]))
