from pathlib import Path
import pandas as pd
import re
import logging

logger = logging.getLogger(__name__)


def handle_duplicate_events(events):
    """Moves the latter event to the next sample."""
    for i, (sample, _, _) in enumerate(events):
        if i > 0 and sample == events[i - 1, 0]:
            events[i, 0] += 1
            logger.info(
                f"Found duplicate event at sample {sample}, moved latter event to sample {events[i, 0]}"
            )
    return events


def get_subjects(root_path):
    return pd.read_csv(Path(root_path) / "participants.tsv", sep="\t")[
        "participant_id"
    ].values


def get_tasks(datapath, subject):
    tasks = set()
    for f in (Path(datapath) / subject / "nirs").glob("*task-*.snirf"):
        fparts = f.stem.split("_")
        for part in fparts:
            if "task" in part:
                task = part.split("-")[1]
                tasks.add(task)
    return list(tasks)


def bids_ch_type_to_mne(ch_type):
    if ch_type == "NIRSCWAMPLITUDE":
        return "fnirs_cw_amplitude"
    elif ch_type == "NIRSCWOPTICALDENSITY":
        return "fnirs_od"
    else:
        raise ValueError(f"ch_type {ch_type} not supported")


def get_ch_type(root_path):
    subjects = get_subjects(root_path)

    # establish expected ch_type from first subject
    subject = subjects[0]
    task = get_tasks(root_path, subject)[0]
    bids_ch_types = pd.read_csv(
        Path(root_path) / subject / "nirs" / f"{subject}_task-{task}_channels.tsv",
        sep="\t",
    )["type"].values
    excepted_ch_type = bids_ch_types[0]

    for subject in subjects:
        tasks = get_tasks(root_path, subject)
        for task in tasks:
            bids_ch_types = pd.read_csv(
                Path(root_path)
                / subject
                / "nirs"
                / f"{subject}_task-{task}_channels.tsv",
                sep="\t",
            )["type"].values
            for ch_type in bids_ch_types:
                if ch_type != excepted_ch_type:
                    raise ValueError(
                        f"found ch_type {ch_type} in {subject} {task} but expected {excepted_ch_type}"
                    )
    return bids_ch_type_to_mne(excepted_ch_type)


def event_mapping_to_task(event_mapping):
    return " v ".join([f'{k.split("/")[-1]}' for k, v in event_mapping.items()])
