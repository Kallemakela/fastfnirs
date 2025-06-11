from collections import defaultdict
import numpy as np
import pandas as pd
import scipy
import scipy.signal
from tqdm import tqdm


class fNIRSFeatureExtractor:
    """
    Extracts features from a 3D array of shape (n_epochs, n_channels, n_samples). Meant to be used as a transformer in a sklearn pipeline.
    """

    def __init__(self, features=["MV"], n_windows=3):
        self.features = features
        self.n_windows = n_windows

    def fit(self, X, y):
        return self

    def transform(self, X):
        f = extract_features_array(X.transpose(0, 2, 1), self.features, self.n_windows)
        return f

    def set_params(self, **params):
        self.features = params["features"]
        self.n_windows = params["n_windows"]
        return self


def extract_freq_features(d, sfreq):
    """
    Extracts frequency domain features for each channel. Implemented according to https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.hilbert.html.

    Parameters
    ----------
    d : numpy.ndarray
        3d array of shape (epochs, chs, samples).
    sfreq : float
        Sampling frequency.

    Returns
    -------
    freq_features : numpy.ndarray
        Frequency domain features for each channel.
    """
    analytic_signal = scipy.signal.hilbert(d, axis=0)
    IA = np.abs(analytic_signal)
    IP = np.unwrap(np.angle(analytic_signal))
    IF = np.diff(IP) / (2.0 * np.pi) * sfreq
    freq_features = np.concatenate([IA, IP, IF], axis=1)
    return freq_features


def extract_features_array(d, features=["MV"], n_windows=3):
    """
    Extracts features from a 3D array of shape (n_epochs, n_samples, n_channels).

    Parameters
    ----------
    x : numpy.ndarray
        Array of shape (n_epochs, n_samples, n_channels).
    """
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
        if "SLO" in features:
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
