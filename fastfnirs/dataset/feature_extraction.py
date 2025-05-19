from collections import defaultdict
import numpy as np
import pandas as pd
import scipy
import scipy.signal
from tqdm import tqdm


def create_windowed_feature_dataset(
    edf, n_windows, chs, tmax, sfreq, include_freq=False, disable_tqdm=False
):
    """
    Creates a windowed feature dataset from epochs dataframe.

    Parameters
    ----------
    edf : pandas.DataFrame
        Epochs dataframe.
    n_windows : int
        Each epoch is divided in n_windows time intervals. E.g. n_windows=2 results in features extracted from 0 to 6 seconds and from 6 to 12 seconds.
    chs : list
        List of channels to include in the feature dataset.
    tmax : float
        End time of epochs.
    sfreq : float
        Sampling frequency in `edf`.
    include_freq : bool, default=False
        Whether to include frequency domain features.
    disable_tqdm : bool, default=False
        Whether to disable tqdm progress bar.

    Returns
    -------
    dataset_df : pandas.DataFrame
        Feature dataset.
    """
    dataset_df = pd.DataFrame(
        columns=["epoch", "condition", "is_bad_epoch", "subject"] + chs
    )
    for e in tqdm(edf["epoch"].unique(), disable=disable_tqdm):
        epoch_df = edf[(edf["epoch"] == e)]
        f_df = extract_window_features_for_epoch(
            epoch_df, chs, l=tmax / n_windows, sfreq=sfreq, include_freq=include_freq
        )
        f_df["epoch"] = e
        f_df[["condition", "is_bad_epoch", "subject"]] = epoch_df.iloc[0][
            ["condition", "is_bad_epoch", "subject"]
        ]
        dataset_df = pd.concat([dataset_df, f_df])
    return dataset_df


def extract_freq_features_for_epoch(epoch_df, chs, sfreq):
    epoch_df = epoch_df[epoch_df["time"] > 0]
    """
    Extracts frequency domain features for each channel. Implemented according to https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.hilbert.html.

    Parameters
    ----------
    epoch_df : pandas.DataFrame
        Epoch dataframe.
    chs : list
        List of channels to extract features for.
    sfreq : float
        Sampling frequency in `edf`.
    
    Returns
    -------
    freq_feature_df : pandas.DataFrame
        Frequency domain features for each channel.
    """
    analytic_signal = scipy.signal.hilbert(epoch_df[chs].T)
    IA = np.abs(analytic_signal)
    IP = np.unwrap(np.angle(analytic_signal))
    IF = np.diff(IP) / (2.0 * np.pi) * sfreq

    freq_feature_columns = (
        ["IA_{:03d}".format(fi) for fi in range(IA.shape[1])]
        + ["IP_{:03d}".format(fi) for fi in range(IP.shape[1])]
        + ["IF_{:03d}".format(fi) for fi in range(IF.shape[1])]
    )
    freq_features = np.concatenate([IA, IP, IF], axis=1)
    freq_feature_df = pd.DataFrame(
        freq_features, index=chs, columns=freq_feature_columns
    )
    freq_feature_df = freq_feature_df.loc[:, sorted(freq_feature_df.columns)].T
    return freq_feature_df


def extract_window_features_for_epoch(epoch_df, chs, l, sfreq, include_freq=False):
    """
    Extracts windowed features for each channel.

    Parameters
    ----------
    epoch_df : pandas.DataFrame
        Epoch dataframe.
    chs : list
        List of channels to extract features for.
    l : float
        Length of each window.
    sfreq : float
        Sampling frequency in `edf`.
    include_freq : bool, default=False
        Whether to include frequency domain features.

    Returns
    -------
    window_feature_df : pandas.DataFrame
        Windowed features for each channel.
    """
    epoch_df = epoch_df[epoch_df["time"] > 0]
    L = int(l * sfreq)  # window length
    n_window = len(epoch_df) // L
    if n_window * L != 12 * 50:
        print("TRIMMING DATA IN FEATURE EXTRACTION")

    window_features_df = pd.DataFrame(index=chs)

    for wi in range(n_window):
        window = epoch_df.iloc[wi * L : (wi + 1) * L, epoch_df.columns.isin(chs)]
        window_features_df["MV_{:03d}".format(wi)] = window.mean(axis=0)
        window_features_df["STD_{:03d}".format(wi)] = window.std(axis=0)
        window_features_df["MAV_{:03d}".format(wi)] = window.abs().mean(axis=0)
        window_features_df["IAV_{:03d}".format(wi)] = window.abs().sum(axis=0)  # ???
        PZN = 0  # how many time signal crosses zero line
        PMN = 0  # how many time signal crosses mean line
        for i in range(len(window) - 1):
            PZN += window.iloc[i] * window.iloc[i + 1] < 0
            PMN += (window.iloc[i] - window_features_df["MV_{:03d}".format(wi)]) * (
                window.iloc[i + 1] - window_features_df["MV_{:03d}".format(wi)]
            ) < 0
        window_features_df["PZN_{:03d}".format(wi)] = PZN
        window_features_df["PMN_{:03d}".format(wi)] = PMN

        deg = 1
        poly_x = np.arange(L)
        polyfit_coef = pd.DataFrame(
            [
                np.polynomial.polynomial.Polynomial.fit(poly_x, window[ch], deg=deg)
                .convert()
                .coef
                for ch in chs
            ],
            index=chs,
            columns=[
                "polyfit_coef_" + str(i) + "_{:03d}".format(wi) for i in range(deg + 1)
            ],
        ).fillna(0.0)
        window_features_df["polyfit_coef_1_{:03d}".format(wi)] = polyfit_coef[
            "polyfit_coef_1_{:03d}".format(wi)
        ]

    """Windowed frequency domain features"""
    if include_freq:
        for wi in range(n_window):
            window = epoch_df.iloc[wi * L : (wi + 1) * L, epoch_df.columns.isin(chs)]
            analytic_signal = scipy.signal.hilbert(window.T)
            IA = np.abs(analytic_signal)
            IP = np.unwrap(np.angle(analytic_signal))
            IF = np.diff(IP) / (2.0 * np.pi) * sfreq
            window_features_df["IA_MV_{:03d}".format(wi)] = IA.mean(axis=1)
            window_features_df["IP_MV_{:03d}".format(wi)] = IP.mean(axis=1)
            window_features_df["IF_MV_{:03d}".format(wi)] = IF.mean(axis=1)
        window_features_df = window_features_df.loc[
            :, sorted(window_features_df.columns)
        ]

    return window_features_df.T


def extract_features_array(d, features=["MV"], n_windows=3):
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
    sXf = np.array(sXf).transpose(1, 2, 0)  # same order as old dataset
    sXf = sXf.reshape(n_epochs, -1)
    return sXf


def extract_features_from_raw(X, features=["MV"], n_windows=3):
    Xf = defaultdict(list)
    for subject, d in X.items():
        Xf[subject] = extract_features_array(d, features=features, n_windows=n_windows)
    return Xf
