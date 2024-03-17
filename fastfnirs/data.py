# %%
import numpy as np
import mne
import warnings
from copy import deepcopy
from fastfnirs.interpolation import interpolate
from joblib import Parallel, delayed
from fastfnirs.utils import load_from, get_cwd, get_cv_from_str, event_mapping_to_task
# %%
def X2grid(X, ch_names, ch2grid, to_grid_params={"h": 5, "w": 11, "c": 1}, **kwargs
):
    print("Converting to grid")

    def X2grid_subject(Xs, ch_names, ch2grid, **kwargs):
        Xs_new = to_grid(Xs, ch_names, ch2grid, **to_grid_params)
        n_epochs, n_chs, h, w, T = Xs_new.shape
        for ei in range(n_epochs):
            for ci in range(n_chs):
                for ti in range(T):
                    # zero_ch_mask = (Xs_new[ei,ci,:,:,ti] == 0)
                    # print(f'{zero_ch_mask.astype(int)}')
                    # if ei % 30 == 0 and ti == T // 2: plot_grid(Xs_new[ei,ci,:,:,ti], title=f'Before interpolation')
                    interpolated = interpolate(Xs_new[ei, ci, :, :, ti], **kwargs)
                    # zero_ch_mask = (interpolated == 0)
                    # print(f'{zero_ch_mask.astype(int)}')
                    # if ei % 30 == 0 and ti == T // 2: plot_grid(interpolated, title=f'After cubic interpolation')
                    if np.any(np.isnan(interpolated)):
                        interpolated = interpolate(interpolated, method="nearest")
                    # zero_ch_mask = (interpolated == 0)
                    # print(f'{zero_ch_mask.astype(int)}')
                    # if ei % 30 == 0 and ti == T // 2: plot_grid(interpolated, title=f'After corner interpolation')
                    Xs_new[ei, ci, :, :, ti] = interpolated
        return Xs_new


def get_ch2grid(chs, w=11, h=5, reverse_y=True):
    """
    Maps channels to a grid of size w x h

    Paremeters
    ----------
    chs: mne.info['chs']
            Channel information
    w: int
            Width of grid
    h: int
            Height of grid
    reverse_y: bool
            Whether to reverse the y axis

    Returns
    -------
    ch2grid: dict
            Dictionary mapping channel names to grid coordinates

    """
    ch2grid = {}
    x_coords = [ch["loc"][0] for ch in chs]
    y_coords = [ch["loc"][1] for ch in chs]
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    x_step = (x_max - x_min) / w
    y_step = (y_max - y_min) / h
    for ch in chs:
        x, y, _ = ch["loc"][:3]
        x_ix = int((x - x_min - 1e-9) / x_step)
        y_ix = int((y - y_min - 1e-9) / y_step)
        if reverse_y:
            y_ix = h - y_ix - 1
        assert x_ix >= 0 and x_ix < w
        assert y_ix >= 0 and y_ix < h
        ch_name = ch["ch_name"]
        ch2grid[ch_name] = (x_ix, y_ix)
    return ch2grid


def sub_dict_split(X, y, train_subjects, test_subjects=None):
    X_train = {k: v for k, v in X.items() if k in train_subjects}
    y_train = {k: v for k, v in y.items() if k in train_subjects}
    if test_subjects is not None and len(test_subjects) > 0:
        X_test = {k: v for k, v in X.items() if k in test_subjects}
        y_test = {k: v for k, v in y.items() if k in test_subjects}
    else:
        X_test, y_test = None, None
    return X_train, y_train, X_test, y_test


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


class BrainDataset:
    def __init__(self, epochs_dict):
        self.epochs_dict = epochs_dict
        self.subjects = list(epochs_dict.keys())
        self.n_epochs = sum([len(epochs_dict[subject]) for subject in self.subjects])
        self.example_epochs = epochs_dict[self.subjects[0]]
        self.all_ch_names = self.example_epochs.ch_names  # not filtered
        self.ch_names = self.all_ch_names  # filtered
        self.event_name_mapping_all = self.example_epochs.event_id
        self.event_name_mapping_task = self.example_epochs.event_id
        self.grid = False
        self.X = None
        self.y = None

    def __len__(self):
        return self.n_epochs

    def __getitem__(self, idx):
        subject = self.subjects[idx // len(self.example_epochs)]  # TODO fix
        epoch = self.epochs_dict[subject][idx % len(self.example_epochs)]
        return epoch

    def get_subject(self, idx):
        return self.subjects[idx]

    def get_subject_epochs(self, idx):
        return self.epochs_dict[self.subjects[idx]]

    def __add__(self, other):
        epochs_dict = {}
        epochs_dict.update(self.epochs_dict)
        epochs_dict.update(other.epochs_dict)
        return BrainDataset(epochs_dict)

    def get_subjects_dataset(self, subjects):
        bds = BrainDataset({subject: self.epochs_dict[subject] for subject in subjects})
        return bds

    def get_full_dataset(self, load=True):
        if load and self.X is not None:
            return self.X, self.y
        X, y = {}, {}
        eventid_to_y = {
            v: self.event_name_mapping_task[k]
            for k, v in self.event_name_mapping_all.items()
            if k in self.event_name_mapping_task
        }
        for subject in self.subjects:
            X[subject], y[subject] = [], []
            subject_epochs = self.epochs_dict[subject]

            for ei in range(len(subject_epochs)):
                epoch = subject_epochs[ei]
                if epoch.events[0, 2] not in eventid_to_y:
                    continue
                # remove everything before 0, and 0
                X[subject].append(
                    epoch.copy().load_data().crop(tmin=0).get_data()[0, :, 1:]
                )
                y[subject].append(eventid_to_y[epoch.events[0, 2]])
            
            if len(X[subject]) > 0:
                X[subject], y[subject] = (
                    np.array(X[subject]),
                    np.array(y[subject]).flatten(),
                )
            else:
                print(f"{subject} no epochs for {event_mapping_to_task(self.event_name_mapping_task)}")
                del X[subject], y[subject]
            
        self.X, self.y = X, y
        return X, y
    
    @property
    def Xc(self):
        return np.concatenate(list(self.X.values()))
    
    @property
    def yc(self):
        return np.concatenate(list(self.y.values()))

    def filter_by_class_count(self, min_count=1):
        """Removes subjects with less than min_count epochs for each class from X and y"""
        n_classes = len(np.unique(self.yc))
        subjects = list(self.X.keys())
        for subject in subjects:
            y = self.y[subject]
            unique, counts = np.unique(y, return_counts=True)
            if np.any(counts < min_count) or len(unique) < n_classes:
                print(f"Removing {subject} due to class count {dict(zip(unique, counts))}")
                del self.X[subject], self.y[subject]
                
        return self.X, self.y


    def downsample(self, T_new=60):
        for sub in self.X.keys():
            T = self.X[sub].shape[-1]
            l = T // T_new
            X_new_sub = np.zeros((*self.X[sub].shape[:-1], T_new))
            for wi in range(T_new):
                ws = wi * l
                we = (wi + 1) * l
                X_new_sub[:, :, wi] = self.X[sub][:, :, ws:we].mean(axis=-1)
            self.X[sub] = X_new_sub
        print(f"Downsampled from {T} to {T_new}")
        return self.X

    def normalize(self, normalize_dims=(0)):
        print(f"Normalizing over dims {normalize_dims}")
        for sub in self.X:
            self.X[sub] = (
                self.X[sub] - self.X[sub].mean(axis=normalize_dims)
            ) / self.X[sub].std(axis=normalize_dims)
        return self.X

    def to_grid(self):
        ch2grid = get_ch2grid(self.example_epochs.info["chs"])
        chs = [ch for ch in self.ch_names if "hbo" in ch]
        self.X = X2grid(self.X, chs, ch2grid)
        self.grid = True
        return self.X

    def apply_ch_selection(self, ch_selection="hbo"):
        print(f"Applying channel selection {ch_selection}")
        if not self.grid:
            ch_mask = np.array([ch_selection in ch for ch in self.ch_names])
            chs_filtered = np.array(self.ch_names)[ch_mask]
            for subject in self.X.keys():
                print(f"Filtering {subject} from {self.X[subject].shape} to {self.X[subject][:, ch_mask, :].shape}")
                self.X[subject] = self.X[subject][:, ch_mask, :]
            self.ch_names = chs_filtered
        else:
            if ch_selection == "hbo":
                for subject in X.keys():
                    Xs = self.X[subject]
                    if len(Xs.shape) == 5:
                        Xs = Xs[:, :1, :, :, :]
                    self.X[subject] = Xs
        return self.X

    def get_split_dataset(self, train_subjects, test_subjects=None):
        X_train, y_train, X_test, y_test = sub_dict_split(
            self.X, self.y, train_subjects, test_subjects
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

    def split(self, train_subjects, test_subjects=None) -> tuple:
        return self.get_subjects_dataset(train_subjects), self.get_subjects_dataset(
            test_subjects
        )

    def process_pipeline(self, verbose=0):
        steps = [
            self.get_full_dataset,
            self.downsample,
            self.normalize,
            self.apply_ch_selection,
        ]
        for step in steps:
            step()
            if verbose > 0:
                X_shape = np.array([self.X[subject].shape for subject in self.subjects])
                print(f"{step.__name__}: {X_shape}")
        return self