"""

"""
import pandas as pd
from pathlib import Path
import logging
import mne
from mne_bids import BIDSPath, read_raw_bids

from fastfnirs.processing import create_epochs_from_raw, process_raw
from fastfnirs.utils import (
    get_ch_type,
    get_subjects,
    get_tasks,
    handle_duplicate_events,
)

logger = logging.getLogger(__name__)


def read_bids_nirs(
    root_path, subject, task, ch_type=None, get_ch_names_map=None, get_montage=None
):
    """
    Reads a subject's nirs data from the BIDS dataset
    """
    subject_id = subject.split("-")[1]
    bidspath = BIDSPath(
        subject=subject_id,
        task=task,
        root=root_path,
        datatype="nirs",
    )
    with mne.utils.use_log_level("ERROR"):
        raw = read_raw_bids(bidspath).load_data()
        if ch_type is not None:
            ch_map = {ch: ch_type for ch in raw.ch_names}
            raw.set_channel_types(ch_map)
        if get_ch_names_map is not None:
            raw.rename_channels(
                get_ch_names_map(root_path, subject, task=task, chs=raw.info["chs"])
            )
        if get_montage is not None:
            raw.set_montage(get_montage(root_path, subject, task))
    return raw


def read_events_metadata(root_path, subject, task):
    """
    Reads a subject's metadata from BIDS and returns it as a dataframe.
    """
    subject_id = subject.split("-")[1]
    sub_events_path = (
        Path(root_path)
        / f"sub-{subject_id}"
        / "nirs"
        / f"sub-{subject_id}_task-{task}_events.tsv"
    )
    metadata = pd.read_csv(sub_events_path, sep="\t")
    na_ix = metadata["onset"].isna()
    if na_ix.any():
        metadata = metadata[~na_ix]
        logger.info(f"Removed {na_ix.sum()} rows with missing onset times.")
    return metadata


def bids_to_mne(
    root_path,
    save_epochs_path=None,
    read_bids_nirs_kwargs={},
    process_raw_kwargs={},
    epochs_kwargs={},
):
    """
    Reads the BIDS data and creates and saves MNE objects in specified formats.

    Parameters
    ----------

    root_path : str
        Path to the BIDS dataset.
    save_epochs_path : str, default: None
        Path to save epochs to. If None, epochs are not saved.

    Returns
    -------
    epochs_dict : dict
        Dictionary of epochs for each subject and task.
    """

    subjects = get_subjects(root_path)
    ch_type = get_ch_type(root_path)
    logger.info(f"ch_type={ch_type}")
    epochs_dict = {}
    for subject in subjects:
        tasks = get_tasks(root_path, subject)
        for task in tasks:
            logger.info(f"Processing subject={subject}, events={task}")
            event_metadata = read_events_metadata(root_path, subject, task=task)
            try:
                raw_cw = read_bids_nirs(
                    root_path,
                    subject,
                    task=task,
                    ch_type=ch_type,
                    **read_bids_nirs_kwargs,
                )
            except OSError as e:
                logger.warning(
                    f"Failed to read {subject} {task}. File probably corrupted."
                )
                print(e)
                # see https://stackoverflow.com/a/43607837 if you want to try to read the file anyway
                continue
            if "fnirs_cw_amplitude" in raw_cw:
                raw_od = mne.preprocessing.nirs.optical_density(raw_cw)
            elif "fnirs_od" in raw_cw:
                raw_od = raw_cw
            raw_haemo = process_raw(raw_od, **process_raw_kwargs)
            events, event_name_mapping = mne.events_from_annotations(raw_haemo)
            events = handle_duplicate_events(events)
            epochs = create_epochs_from_raw(
                raw_haemo,
                events=events,
                event_metadata=event_metadata,
                event_name_mapping=event_name_mapping,
                **epochs_kwargs,
            )
            if save_epochs_path is not None:
                epochs_path = Path(save_epochs_path)
                epochs.save(
                    epochs_path / f"{subject}_task-{task}_epo.fif",
                    overwrite=True,
                    verbose="WARNING",
                )
                logger.info(
                    f'Saved epochs to {epochs_path/f"{subject}_task-{task}_epo.fif"}'
                )
            if task not in epochs_dict:
                epochs_dict[task] = {}
            epochs_dict[task][subject] = epochs
    return epochs_dict
