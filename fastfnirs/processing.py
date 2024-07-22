import numpy as np
import pandas as pd
import mne
from itertools import compress
import warnings


def interpolate_bads_nirs(inst, method="nearest", exclude=(), verbose=None):
    """
    Added method='average_nearest' to this mne function. It takes the average of the nearest instead of picking the first.
    """
    from scipy.spatial.distance import pdist, squareform
    from mne.preprocessing.nirs import _channel_frequencies, _check_channels_ordered

    # Returns pick of all nirs and ensures channels are correctly ordered
    freqs = np.unique(_channel_frequencies(inst.info))
    picks_nirs = _check_channels_ordered(inst.info, freqs)
    picks_nirs = sorted(
        picks_nirs
    )  # in new versions of mne, _check_channels_ordered returns channels in different order than data
    if len(picks_nirs) == 0:
        return

    nirs_ch_names = [inst.info["ch_names"][p] for p in picks_nirs]
    nirs_ch_names = [ch for ch in nirs_ch_names if ch not in exclude]
    bads_nirs = [ch for ch in inst.info["bads"] if ch in nirs_ch_names]
    if len(bads_nirs) == 0:
        return
    picks_bad = mne.io.pick.pick_channels(inst.info["ch_names"], bads_nirs, exclude=[])
    bads_mask = [p in picks_bad for p in picks_nirs]

    chs = [inst.info["chs"][i] for i in picks_nirs]
    locs3d = np.array([ch["loc"][:3] for ch in chs])

    mne.utils._check_option("fnirs_method", method, ["nearest", "average_nearest"])

    if method == "nearest":
        dist = pdist(locs3d)
        dist = squareform(dist)

        for bad in picks_bad:
            dists_to_bad = dist[bad]
            # Ignore distances to self
            dists_to_bad[dists_to_bad == 0] = np.inf
            # Ignore distances to other bad channels
            dists_to_bad[bads_mask] = np.inf
            # Find closest remaining channels for same frequency
            closest_idx = np.argmin(dists_to_bad) + (bad % 2)
            inst._data[bad] = inst._data[closest_idx]

        inst.info["bads"] = [ch for ch in inst.info["bads"] if ch in exclude]

    elif method == "average_nearest":
        """
        Takes mean of all nearest channels instead of just one
        """

        dist = pdist(locs3d)
        dist = squareform(dist)

        for bad in picks_bad:
            dists_to_bad = dist[bad]
            # Ignore distances to self
            dists_to_bad[dists_to_bad == 0] = np.inf
            # Ignore distances to other bad channels
            dists_to_bad[bads_mask] = np.inf
            # Find closest remaining channels
            all_closest_idxs = np.argwhere(
                np.isclose(dists_to_bad, np.min(dists_to_bad))
            )
            # Filter for same frequency as bad
            all_closest_idxs = all_closest_idxs[all_closest_idxs % 2 == bad % 2]
            inst._data[bad] = np.mean(inst._data[all_closest_idxs], axis=0)

        inst.info["bads"] = [ch for ch in inst.info["bads"] if ch in exclude]

    return inst


def remove_short_channels(raw, min_length=0.01):
    picks = mne.pick_types(raw.info, meg=False, fnirs=True)
    dists = mne.preprocessing.nirs.source_detector_distances(raw.info, picks=picks)
    raw.pick(picks[dists > min_length])
    return raw


def process_raw(
    raw: mne.io.Raw,
    ch_interpolation="interpolate_nearest",
    sci_threshold=0.5,
    tddr=True,
    l_freq=0.01,
    l_trans_bandwidth=0.004,
    h_freq=0.1,
    h_trans_bandwidth=0.01,
    bll_ppf=6,
    filter_length="auto",
    filter_kwargs={},
    remove_short=None,
    verbose=False,
):
    """
    Applies preprocesing to a raw object.

    1. Detects bad channels with SCI and interpolates them. Use `ch_interpolation`=None to not interpolate.
    2. Applies TDDR to the raw object.
    3. Converts the optical density data to haemoglobin concentration using the modified Beer-Lambert law.
    4. Applies a band-pass (l_freq=0.01, h_freq=0.1) filter to the raw object
    Parameters

    ----------
    file_path : str
            Path to the file to read.
    data_type : str
            Type of data to read. Only 'OD' is currently supported.
    ch_interpolation : str
            Method to use to for channel interpolation.
    include_events : str or list, default 'empe'
            Events to include. Codes are defined in data/triggers.txt. Only 'empe' is currently supported.
    tddr : bool, default True
            Whether to apply Temporal Derivative Distribution Repair (TDDR).
    sci_threshold : float, default 0.5
            Threshold for the Scalp Coupling Index (SCI).
    l_freq : float, default 0.01
            Low cut-off frequency for the band-pass filter.
    h_freq : float, default 0.1
            High cut-off frequency for the band-pass filter.
    bll_ppf : int, default 6
            PPF for the modified Beer-Lambert law.
    filter_length : str or int, default 'auto'
            Length of the FIR filter to use. If 'auto', the filter length is chosen based on the sampling frequency.
    verbose : bool, default False
            Whether to print progress.
    remove_short_channels : float, default None
            Remove channels with source-detector distance less than this value.
    """

    if remove_short is not None:
        raw = remove_short_channels(raw, min_length=remove_short)

    raw = process_raw_od(
        raw,
        ch_interpolation=ch_interpolation,
        sci_threshold=sci_threshold,
        tddr=tddr,
        verbose=verbose,
    )
    # best practice is to use ppf=6, see https://github.com/mne-tools/mne-python/pull/9843
    raw = mne.preprocessing.nirs.beer_lambert_law(raw, ppf=bll_ppf)
    raw = raw.filter(
        l_freq=l_freq,
        h_freq=h_freq,
        l_trans_bandwidth=l_trans_bandwidth,
        h_trans_bandwidth=h_trans_bandwidth,
        filter_length=filter_length,
        verbose=False,
        **filter_kwargs,
    )
    return raw


def process_raw_od(
    raw: mne.io.Raw,
    ch_interpolation="interpolate_nearest",
    sci_threshold=0.5,
    tddr=True,
    verbose=False,
) -> mne.io.Raw:
    with np.errstate(
        invalid="ignore"
    ):  # some channels have all zeros, they will be eliminated
        sci = mne.preprocessing.nirs.scalp_coupling_index(raw)
    raw.info["bads"] = list(
        compress(raw.ch_names, (np.isnan(sci) | (sci <= sci_threshold)))
    )
    if verbose:
        print(f'{len(raw.info["bads"])}/{len(raw.ch_names)} channels marked as bad')

    if ch_interpolation == "interpolate_average_nearest":
        interpolate_bads_nirs(raw, method="average_nearest")
    elif ch_interpolation == "interpolate_nearest":
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            raw.interpolate_bads(verbose=verbose)
    elif ch_interpolation == "drop":
        raw.drop_channels(raw.info["bads"])

    if tddr:
        raw = mne.preprocessing.nirs.temporal_derivative_distribution_repair(raw)

    return raw


def create_epochs_from_raw(
    raw,
    events=None,
    event_name_mapping=None,
    event_metadata=None,
    tmin=-5,
    tmax=12,
    reject_criteria=None,
    verbose=False,
    **kwargs,
):
    """
    Creates preprocessed MNE epoch objects for a subject.

    1. Marks bad epochs based on maximum peak-to-peak signal amplitude (PTP)
    2. Saves `is_bad_epoch` and `bad_epoch_reason` to epochs.metadata.
    3. Removes epochs with missing data.

    Parameters
    ----------
    raw : mne.io.Raw
            Processed MNE raw object.
    events : array
            Events array.
    event_name_mapping : dict
            Mapping from event names to event codes.
    event_metadata : pd.DataFrame
            Event metadata.
    tmin : float, default -5
            Time before event to include in epoch.
    tmax : float, default 12
            Time after event to include in epoch.
    reject_criteria : dict, default dict(hbo=80e-6)
            Criteria for rejecting epochs. Keys are channel types and values are the maximum PTP.
    verbose : bool, default False
            Whether to print progress.

    Returns
    -------
    epochs : mne.Epochs
            The epochs object.
    """
    subject = raw.info["subject_info"]["his_id"]
    subject = subject if subject.startswith("sub-") else f"sub-{subject}"

    if events is None or event_name_mapping is None:
        events, event_name_mapping = mne.events_from_annotations(
            raw, event_id=event_name_mapping, verbose=verbose
        )

    if event_metadata is None:
        event_metadata = pd.DataFrame(events[:, ::2], columns=["onset", "value"])

    event_metadata["subject"] = subject
    epochs = mne.Epochs(
        raw,
        events,
        event_id=event_name_mapping,
        metadata=event_metadata,
        tmin=tmin,
        tmax=tmax,
        verbose=False,
        **kwargs,
    )

    # only mark bad epochs since mne deletes bad epochs on save
    epochs.metadata["bad_epoch_reason"] = (
        epochs.copy().drop_bad(reject=reject_criteria, verbose=verbose).drop_log
    )
    epochs.metadata["is_bad_epoch"] = (
        epochs.metadata["bad_epoch_reason"].astype(str) != "()"
    )
    epochs.metadata["bad_channels"] = list(
        np.tile(epochs.info["bads"], (len(epochs.metadata), 1))
    )
    epochs.drop_bad(verbose=verbose)  # removes epochs with missing data
    return epochs
