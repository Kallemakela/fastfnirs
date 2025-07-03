import numpy as np
import mne
from itertools import compress


def fill_missing_chs(raw, all_chs):
    """Adds emtpy channels to raw data if they are missing. Marks them as bad so they can be interpolated later."""
    missing_chs = set(all_chs) - set(raw.ch_names)
    if missing_chs:
        print(f"Adding missing channels: {missing_chs}")
        for ch in missing_chs:
            missing_ch_info = mne.create_info(
                ch_names=[ch], sfreq=raw.info["sfreq"], ch_types=[ch.split(" ")[1]]
            )
            missing_ch_data = mne.io.RawArray(
                np.zeros((1, raw.n_times)), missing_ch_info, verbose=False
            )
            raw.add_channels([missing_ch_data], force_update_info=True)
            raw.info["bads"].append(ch)
    return raw


def apply_sci(raw, sci_threshold, verbose=False):
    """Helper function to mark channels as bad based on the scalp coupling index (SCI)."""
    with np.errstate(
        invalid="ignore"
    ):  # some channels have all zeros, they will be eliminated
        sci = scalp_coupling_index(raw)
    below_threshold = np.isnan(sci) | (sci <= sci_threshold)
    raw.info["bads"] = list(compress(raw.ch_names, below_threshold))
    elim_ratio = np.sum(below_threshold) / len(below_threshold)
    if elim_ratio > 0.5:
        print(f"Warning: {elim_ratio:.2f} channels eliminated due to low SCI")
    if verbose:
        print(f'{len(raw.info["bads"])}/{len(raw.ch_names)} channels marked as bad')
    return raw


def scalp_coupling_index(
    raw,
    l_freq=0.7,
    h_freq=1.5,
    l_trans_bandwidth=0.3,
    h_trans_bandwidth=0.3,
    verbose=False,
):
    r"""

    Copied from MNE. Removed ch type checks.

    Authors: Robert Luke <mail@robertluke.net>
             Eric Larson <larson.eric.d@gmail.com>
             Alexandre Gramfort <alexandre.gramfort@inria.fr>

    License: BSD-3-Clause

    This function calculates the scalp coupling index
    :footcite:`pollonini2014auditory`. This is a measure of the quality of the
    connection between the optode and the scalp.

    Parameters
    ----------
    raw : instance of Raw
        The raw data.
    %(l_freq)s
    %(h_freq)s
    %(l_trans_bandwidth)s
    %(h_trans_bandwidth)s
    %(verbose)s

    Returns
    -------
    sci : array of float
        Array containing scalp coupling index for each channel.

    References
    ----------
    .. footbibliography::
    """
    raw = raw.copy().load_data()
    zero_mask = np.std(raw._data, axis=-1) == 0
    filtered_data = raw.filter(
        l_freq,
        h_freq,
        l_trans_bandwidth=l_trans_bandwidth,
        h_trans_bandwidth=h_trans_bandwidth,
        verbose=verbose,
    ).get_data()

    sci = np.zeros(raw._data.shape[0])
    ch_names = raw.ch_names
    ch_sd = [ch.split(" ")[0] for ch in ch_names]
    for loc in ch_sd:
        loc_chs = [i for i, ch in enumerate(ch_names) if ch.startswith(loc)]
        if len(loc_chs) != 2:
            raise ValueError(
                f"Expected two channels for each location, found {len(loc_chs)} for {loc}"
            )
        ci1, ci2 = loc_chs
        with np.errstate(invalid="ignore"):
            c = np.corrcoef(filtered_data[ci1], filtered_data[ci2])[0][1]
        if not np.isfinite(c):  # someone had std=0
            c = 0
        sci[ci1] = c
        sci[ci2] = c
    sci[zero_mask] = 0
    return sci


def temporal_derivative_distribution_repair(raw):
    """
    Function taken from MNE-Python and modified to work on non-standard
    channels

    # Authors: Robert Luke <mail@robertluke.net> and Frank Fishburn
    # License: BSD-3-Clause

    Apply temporal derivative distribution repair to data.

    Applies temporal derivative distribution repair (TDDR) to data
    (Fishburn et al. 2019). This approach removes baseline shift
    and spike artifacts without the need for any user-supplied parameters.

    Parameters
    ----------
    raw : instance of Raw
        The raw data.

    Returns
    -------
    raw : instance of Raw
         Data with TDDR applied.
    """
    from mne.io import BaseRaw
    from mne.utils import _validate_type
    from mne.io.pick import _picks_to_idx

    raw = raw.copy().load_data()
    _validate_type(raw, BaseRaw, "raw")

    picks = _picks_to_idx(raw.info, "fnirs", exclude=[])
    for pick in picks:
        raw._data[pick] = _TDDR(raw._data[pick], raw.info["sfreq"])

    return raw


def _TDDR(signal, sample_rate):
    """
    Function taken from MNE-Python

    # Authors: Robert Luke <mail@robertluke.net> and Frank Fishburn
    # License: BSD-3-Clause
    """
    # This function is the reference implementation for the TDDR algorithm for
    #   motion correction of fNIRS data, as described in:
    #
    #   Fishburn F.A., Ludlum R.S., Vaidya C.J., & Medvedev A.V. (2019).
    #   Temporal Derivative Distribution Repair (TDDR): A motion correction
    #   method for fNIRS. NeuroImage, 184, 171-179.
    #   https://doi.org/10.1016/j.neuroimage.2018.09.025
    #
    # Usage:
    #   signals_corrected = TDDR( signals , sample_rate );
    #
    # Inputs:
    #   signals: A [sample x channel] matrix of uncorrected optical density or
    #            hb data
    #   sample_rate: A scalar reflecting the rate of acquisition in Hz
    #
    # Outputs:
    #   signals_corrected: A [sample x channel] matrix of corrected optical
    #   density data
    from scipy.signal import butter, filtfilt

    signal = np.array(signal)
    if len(signal.shape) != 1:
        for ch in range(signal.shape[1]):
            signal[:, ch] = _TDDR(signal[:, ch], sample_rate)
        return signal

    # Preprocess: Separate high and low frequencies
    filter_cutoff = 0.5
    filter_order = 3
    Fc = filter_cutoff * 2 / sample_rate
    signal_mean = np.mean(signal)
    signal -= signal_mean
    if Fc < 1:
        fb, fa = butter(filter_order, Fc)
        signal_low = filtfilt(fb, fa, signal, padlen=0)
    else:
        signal_low = signal

    signal_high = signal - signal_low

    # Initialize
    tune = 4.685
    D = np.sqrt(np.finfo(signal.dtype).eps)
    mu = np.inf
    iter = 0

    # Step 1. Compute temporal derivative of the signal
    deriv = np.diff(signal_low)

    # Step 2. Initialize observation weights
    w = np.ones(deriv.shape)

    # Step 3. Iterative estimation of robust weights
    while iter < 50:

        iter = iter + 1
        mu0 = mu

        # Step 3a. Estimate weighted mean
        mu = np.sum(w * deriv) / np.sum(w)

        # Step 3b. Calculate absolute residuals of estimate
        dev = np.abs(deriv - mu)

        # Step 3c. Robust estimate of standard deviation of the residuals
        sigma = 1.4826 * np.median(dev)

        # Step 3d. Scale deviations by standard deviation and tuning parameter
        if sigma == 0:
            break
        r = dev / (sigma * tune)

        # Step 3e. Calculate new weights according to Tukey's biweight function
        w = ((1 - r**2) * (r < 1)) ** 2

        # Step 3f. Terminate if new estimate is within
        # machine-precision of old estimate
        if abs(mu - mu0) < D * max(abs(mu), abs(mu0)):
            break

    # Step 4. Apply robust weights to centered derivative
    new_deriv = w * (deriv - mu)

    # Step 5. Integrate corrected derivative
    signal_low_corrected = np.cumsum(np.insert(new_deriv, 0, 0.0))

    # Postprocess: Center the corrected signal
    signal_low_corrected = signal_low_corrected - np.mean(signal_low_corrected)

    # Postprocess: Merge back with uncorrected high frequency component
    signal_corrected = signal_low_corrected + signal_high + signal_mean

    return signal_corrected
