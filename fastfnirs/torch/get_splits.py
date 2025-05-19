import numpy as np

from fastfnirs.utils import get_cv_from_str
from fastfnirs.torch.torch_utils import (
    sub_dict_split,
    sub_dict_to_tensor,
    create_dataset,
)


def get_split_dataset(X, y, train_subjects, test_subjects=None):
    X_train, y_train, X_test, y_test = sub_dict_split(
        X, y, train_subjects, test_subjects
    )
    X_train, y_train, subject_ix_train = sub_dict_to_tensor(X_train, y_train)
    info = {
        "subject_ix_train": subject_ix_train,
    }
    if test_subjects is not None:
        X_test, y_test, subject_ix_test = sub_dict_to_tensor(X_test, y_test)
        info["subject_ix_test"] = subject_ix_test
    else:
        X_test, y_test = None, None
    return X_train, y_train, X_test, y_test, info


def get_train_val_test_dataloaders(
    bd, cv_str, split_ix, seed, include_val_set=True, batch_size=32, num_workers=8
):
    # Initial train-validation and test split
    subjects = np.array(bd.subjects)
    splits = list(get_cv_from_str(cv_str, seed=seed).split(subjects))
    split = splits[split_ix]
    train_val_subjects_ix, test_subjects_ix = split
    train_val_subjects = subjects[train_val_subjects_ix]
    test_subjects = subjects[test_subjects_ix]

    # Sub split the train-validation set to train and validation sets
    val_cv = get_cv_from_str(f"k{int(cv_str[1:])-1}", seed=seed)
    train_subjects_ix, val_subjects_ix = list(val_cv.split(train_val_subjects))[0]
    train_subjects = train_val_subjects[train_subjects_ix]
    val_subjects = train_val_subjects[val_subjects_ix]

    if include_val_set:
        # Sub split the train-validation set to train and validation sets
        val_cv = get_cv_from_str(f"k{int(cv_str[1:])-1}", seed=seed)
        train_subjects_ix, val_subjects_ix = list(val_cv.split(train_val_subjects))[0]
        train_subjects = train_val_subjects[train_subjects_ix]
        val_subjects = train_val_subjects[val_subjects_ix]
    else:
        train_subjects = train_val_subjects
        val_subjects = np.array([])  # Empty array for consistency

    # Assertions to ensure no intersection between sets
    assert len(set(train_subjects).intersection(set(val_subjects))) == 0
    assert len(set(train_subjects).intersection(set(test_subjects))) == 0
    assert len(set(val_subjects).intersection(set(test_subjects))) == 0

    print(
        f"Train subjects: {len(train_subjects)} Val subjects: {len(val_subjects)} Test subjects: {len(test_subjects)}"
    )
    print(f"{list(train_subjects)=}\n{list(val_subjects)=}\n{list(test_subjects)=}")

    X_, y_, X_test, y_test = sub_dict_split(
        bd.X, bd.y, train_val_subjects, test_subjects
    )
    X_train, y_train, X_val, y_val = sub_dict_split(
        X_, y_, train_subjects, val_subjects
    )

    X_train, y_train, _ = sub_dict_to_tensor(X_train, y_train)
    X_test, y_test, _ = sub_dict_to_tensor(X_test, y_test)

    print(f"{X_train.shape=}, y={np.unique(y_train, return_counts=True)}")
    print(f"{X_test.shape=}, y={np.unique(y_test, return_counts=True)}")

    train_dataset = create_dataset(X_train, y_train)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )

    test_dataset = create_dataset(X_test, y_test)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    loaders = {"train": train_loader, "test": test_loader}

    if include_val_set:
        X_val, y_val, _ = sub_dict_to_tensor(X_val, y_val)
        val_dataset = create_dataset(X_val, y_val)
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )
        print(f"{X_val.shape=}, y={np.unique(y_val, return_counts=True)}")
        loaders["val"] = val_loader

    return loaders
