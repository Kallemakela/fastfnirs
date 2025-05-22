from pathlib import Path
import numpy as np
import mne
import pandas as pd
from fastfnirs.dataset.feature_extraction import extract_features_from_raw
from fastfnirs.dataset.grid import X2grid_new, get_ch2grid


def remove_chs_below_threshold(epochs_dict, threshold=-0.07):
    return {
        k: v.pick([c["ch_name"] for c in v.info["chs"] if c["loc"][1] > threshold])
        for k, v in epochs_dict.items()
    }


class BrainDataset:
    def __init__(self, epochs_dict):
        self.epochs_dict = epochs_dict
        self.subjects = np.array(list(epochs_dict.keys()))
        self.n_epochs = sum([len(epochs_dict[subject]) for subject in self.subjects])
        self.example_epochs = epochs_dict[self.subjects[0]]
        self.all_ch_names = self.example_epochs.ch_names  # not filtered
        self.ch_names = self.all_ch_names  # filtered
        self.event_name_mapping_all = self.example_epochs.event_id
        self.event_name_mapping_task = self.example_epochs.event_id
        self.grid = False
        self.X = None
        self.y = None
        self.sfreq = self.example_epochs.info["sfreq"]

    def __len__(self):
        return self.n_epochs

    def __getitem__(self, idx):
        i = 0
        for subject in self.subjects:
            epochs = self.epochs_dict[subject]
            for ei in range(len(epochs)):
                epoch = epochs[ei]
                if i == idx:
                    return epoch
                i += 1
        return epoch

    def get_all_epochs(self):
        return mne.concatenate_epochs(
            [self.epochs_dict[subject] for subject in self.subjects]
        )

    @property
    def Xc(self):
        return np.concatenate(list(self.X.values()))

    @property
    def yc(self):
        return np.concatenate(list(self.y.values()))

    @property
    def groups(self):
        return np.concatenate(
            [[i] * len(self.X[subject]) for i, subject in enumerate(self.X.keys())]
        )

    @property
    def n_classes(self):
        return len(np.unique(self.yc))

    def get_subject(self, idx):
        return self.subjects[idx]

    def get_subject_epochs(self, idx):
        return self.epochs_dict[self.subjects[idx]]

    def __add__(self, other):
        epochs_dict = {}
        epochs_dict.update(self.epochs_dict)
        epochs_dict.update(other.epochs_dict)
        return BrainDataset(epochs_dict)

    def load_epoch_data(self):
        for subject in self.subjects:
            self.epochs_dict[subject].load_data()
        return self

    def get_metadata(self):
        metadata = []
        for subject in self.subjects:
            metadata.append(self.epochs_dict[subject].metadata)
        return pd.concat(metadata)

    def get_subjects_dataset(self, subjects):
        bds = BrainDataset({subject: self.epochs_dict[subject] for subject in subjects})
        return bds

    def get_full_dataset(self, load=True, **kwargs):
        """Converts epochs_dict to self.X, self.y"""
        if load and self.X is not None:
            return self.X, self.y
        X, y = {}, {}
        eventid_to_y = {
            v: self.event_name_mapping_task[k]
            for k, v in self.event_name_mapping_all.items()
            if k in self.event_name_mapping_task
        }
        task_trial_types = list(self.event_name_mapping_task.keys())
        for subject in self.subjects:
            X[subject], y[subject] = [], []
            subject_epochs = self.epochs_dict[subject][task_trial_types]
            for ei in range(len(subject_epochs)):
                epoch = subject_epochs[ei]
                # remove everything before 0, and 0
                X[subject].append(
                    epoch.copy().load_data().crop(tmin=0).get_data()[0, :, 1:]
                )
                y[subject].append(eventid_to_y[epoch.events[0, 2]])
            X[subject], y[subject] = (
                np.array(X[subject]),
                np.array(y[subject]).flatten(),
            )
        self.X, self.y = X, y
        return self

    def separate_ch_types(self, **kwargs):
        """
        Separates ch types in self.X to e.g. hbo and hbr.

        (epochs, chs, T) -> (epochs, ch_types, chs/ch_types, T)
        """
        ch_types = [c.split()[-1] for c in self.ch_names]
        ch_types_unique = np.sort(np.unique(ch_types))
        X_new = {}
        for subject in self.X:
            d = self.X[subject].shape
            X_new[subject] = np.zeros(
                (*d[:-2], len(ch_types_unique), d[-2] // len(ch_types_unique), d[-1])
            )
            for cti, ct in enumerate(ch_types_unique):
                ch_mask = np.array([ct in ch for ch in self.ch_names])
                X_new[subject][:, cti] = self.X[subject][:, ch_mask]
        self.X = X_new
        return self

    def channels_separated(self, **kwargs) -> bool:
        ch_types_unique = np.unique([c.split()[-1] for c in self.ch_names])
        return self.X.shape[1] == len(ch_types_unique)

    def extract_features(self, **kwargs):
        X = {sub: self.X[sub].transpose(0, 2, 1) for sub in self.X}
        self.X = extract_features_from_raw(
            X,
            **kwargs,
        )
        return self

    def downsample(self, T_new=60, **kwargs):
        for sub in self.X.keys():
            T = self.X[sub].shape[-1]
            l = T // T_new
            X_new_sub = np.zeros((*self.X[sub].shape[:-1], T_new))
            for wi in range(T_new):
                ws = wi * l
                we = (wi + 1) * l
                X_new_sub[:, :, wi] = self.X[sub][:, :, ws:we].mean(axis=-1)
            self.X[sub] = X_new_sub
        self.sfreq = self.sfreq // l
        print(f"Downsampled from {T} to {T_new}, sfreq: {self.sfreq:.1f}")
        return self

    def normalize(self, normalize_dims=(0, 1, 2), **kwargs):
        if normalize_dims is None or None in normalize_dims:
            return self
        print(f"Normalizing over dims {normalize_dims}")
        for sub in self.X:
            mu = self.X[sub].mean(normalize_dims, keepdims=True)
            std = self.X[sub].std(normalize_dims, keepdims=True)
            self.X[sub] = (self.X[sub] - mu) / std
        return self

    def to_grid(self, **kwargs):
        self.ch2grid = get_ch2grid(self.example_epochs.info["chs"], **kwargs)
        chs = [ch for ch in self.ch_names if "hbo" in ch]
        self.X = X2grid_new(self.X, chs, self.ch2grid, **kwargs)
        # for individual ch2grid for each subject use: self.X = bd2grid_ind(bd)
        self.grid = True
        return self

    def apply_ch_selection(self, ch_selection="hbo", **kwargs):
        # print(f'Applying channel selection {ch_selection}')
        if ch_selection == "all":
            return self

        if not self.grid:
            ch_mask = np.array([ch_selection in ch for ch in self.ch_names])
            chs_filtered = np.array(self.ch_names)[ch_mask]
            for subject in self.X.keys():
                self.X[subject] = self.X[subject][:, ch_mask, :]
            self.ch_names = chs_filtered
        else:
            if ch_selection == "hbo":
                for subject in self.X.keys():
                    Xs = self.X[subject]
                    if len(Xs.shape) == 5:
                        Xs = Xs[:, :1, :, :, :]
                    self.X[subject] = Xs
        return self

    def split(self, train_subjects, test_subjects=None) -> tuple:
        return self.get_subjects_dataset(train_subjects), self.get_subjects_dataset(
            test_subjects
        )

    def filter_by_class_count(self, min_count=1):
        """Removes subjects with less than min_count epochs for each class from X and y"""
        n_classes = len(np.unique(self.yc))
        subjects = list(self.X.keys())
        for subject in subjects:
            y = self.y[subject]
            unique, counts = np.unique(y, return_counts=True)
            if np.any(counts < min_count) or len(unique) < n_classes:
                if self.verbose > 0:
                    print(
                        f"Removing {subject} due to class count {dict(zip(unique, counts))}"
                    )
                del self.X[subject], self.y[subject]

        return self.X, self.y

    def keep_classes(self, classes):
        """Keeps only the classes in classes"""
        for subject in self.X.keys():
            ys = self.y[subject]
            mask = np.isin(ys, list(classes))
            self.X[subject] = self.X[subject][mask]
            self.y[subject] = self.y[subject][mask]
        return self

    def process_pipeline(self, verbose=0, **kwargs):
        steps = [
            self.get_full_dataset,
            self.downsample,
            self.normalize,
            self.apply_ch_selection,
        ]
        for step in steps:
            step(**kwargs)
            if verbose > 0:
                X_shape = np.array([self.X[subject].shape for subject in self.subjects])
                print(f"{step.__name__}: {X_shape}")
        return self

    @staticmethod
    def load_bd(epochs_dict, event_name_mapping=None, **kwargs):
        brain_dataset = BrainDataset(epochs_dict)
        brain_dataset.event_name_mapping_task = event_name_mapping
        return brain_dataset

    @staticmethod
    def load_data_2d(dataset_name, **kwargs):
        brain_dataset = BrainDataset.load_bd(dataset_name, **kwargs)
        brain_dataset.process_pipeline(**kwargs)
        return brain_dataset

    @staticmethod
    def load_data_grid(dataset_name, **kwargs):
        brain_dataset = BrainDataset.load_data_2d(dataset_name, **kwargs)
        brain_dataset.to_grid()
        return brain_dataset
