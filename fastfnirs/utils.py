from pathlib import Path
import numpy as np
import pandas as pd
import re
import logging
from sklearn.metrics import accuracy_score
from scipy.stats import bootstrap

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


def divide_to_groups(l, n):
    return [l[i : i + n] for i in range(0, len(l), n)]


def get_cwd():
    return Path(__file__).parents[1]


def save_to(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_from(path):
    with open(path, "rb") as f:
        obj = pickle.load(f)
    return obj


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def get_scorer_with_kwargs(scorer, **kwargs):
    def scorer_with_kwargs(y_true, y_pred):
        return scorer(y_true, y_pred, **kwargs)

    return scorer_with_kwargs


def format_subject(subject):
    if isinstance(subject, int):
        return f"sub-{subject:d}"
    elif isinstance(subject, str):
        if subject.startswith("sub-"):
            return subject
        else:
            return f"sub-{subject}"
    else:
        raise ValueError(f"Unknown subject format: {subject}")


def eval_defautdict(s):
    p = re.compile(r"^defaultdict\(<class '(\w+)'>")
    c = p.findall(s)[0]
    new_s = s.replace("<class '%s'>" % c, c)
    return eval(new_s)


def bootstrap_eval(y, y_pred, statistic=accuracy_score, n_resamples=10000, seed=0):
    return bootstrap(
        (y, y_pred),
        paired=True,
        statistic=statistic,
        vectorized=False,
        method="percentile",
        n_resamples=10000,
        random_state=seed,
    )


def bres_str(bres):
    ci_l, ci_u = bres.confidence_interval
    se = bres.standard_error
    return f"± {se:0.4f} ({ci_l:0.4f}, {ci_u:0.4f})"


def get_cv_from_str(cv_str, n=None, y=None, seed=None, **kwargs):
    if re.match(r"k\d+", cv_str):
        from sklearn.model_selection import KFold

        k = int(cv_str[1:])
        if seed is None:
            cv = KFold(n_splits=k)
        else:
            cv = KFold(n_splits=k, shuffle=True, random_state=seed)
    elif cv_str == "loo":
        from sklearn.model_selection import KFold

        cv = KFold(n_splits=n)
    elif cv_str == "looc":
        from sklearn.model_selection import RepeatedStratifiedKFold

        _, label_counts = np.unique(y, return_counts=True)
        if seed == None:
            seed = 1
        cv = RepeatedStratifiedKFold(
            n_splits=np.min(label_counts), n_repeats=1, random_state=seed, **kwargs
        )
    return cv


def find_lcs(strings):
    """Find the longest common substring of a list of strings."""
    # Step 1: Find the shortest string in the list.
    shortest = min(strings, key=len)
    
    # Step 2: Generate all substrings of the shortest string, from longest to shortest.
    for length in range(len(shortest), 0, -1):
        for start in range(len(shortest) - length + 1):
            substr = shortest[start:start + length]
            
            # Step 3: Check if this substring is common to all strings.
            if all(substr in string for string in strings):
                return substr
    return ""  # Return an empty string if there is no common substring.


def remove_common_part(event_mapping):
    """remove common part from event names"""
    substr_all = find_lcs(list(event_mapping.keys()))
    return {k.replace(substr_all, ""): v for k, v in event_mapping.items()}


def combine_event_map_by_substr(event_mapping):
    """
    Combines event names by finding the longest common substring.

    e.g. {'a1': 0, 'a2': 0, 'b1': 1, 'b2': 1} --> {'a': 0, 'b': 1}
    """
    pretty = {}
    event_mapping_ = event_mapping.copy()
    unique_vals = np.unique(list(event_mapping.values()))

    for u in unique_vals:
        val_keys = [k for k, v in event_mapping_.items() if v == u]
        common_part = find_lcs(val_keys)
        if common_part in pretty:
            return combine_event_map_by_substr(remove_common_part(event_mapping))
        if len(common_part) == 0:
            raise ValueError(f"Could not find common part for {val_keys}")
        pretty[common_part] = u
    return pretty


def combine_event_map_by_concat(event_mapping):
    """
    Concatenates event names by combining them with a '&'.
    """
    event_mapping_ = {}
    for k, v in event_mapping.items():
        if v not in event_mapping_:
            event_mapping_[v] = k
        else:
            event_mapping_[v] += f" & {k}"
    return {v: k for k, v in event_mapping_.items()}


def combine_event_map(event_mapping):
    """
    Combines event names by finding the longest common substring.

    e.g. {'a1': 0, 'a2': 0, 'b1': 1, 'b2': 1} --> {'a': 0, 'b': 1}

    If can't find common part, concatenates event names with a '&'.
    """
    try:
        return combine_event_map_by_substr(event_mapping)
    except ValueError:
        return combine_event_map_by_concat(event_mapping)


def event_mapping_to_task(event_mapping, combine=True):
    if combine:
        event_mapping = combine_event_map(event_mapping)
    return " V ".join([f'{k.split("/")[-1]}' for k, v in event_mapping.items()])