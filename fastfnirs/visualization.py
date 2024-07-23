import mne
import matplotlib.pyplot as plt
from fastfnirs.utils import combine_event_map
from pathlib import Path


def plot_evoked(epochs_dict, **kwargs):
    all_epochs = mne.concatenate_epochs(list(epochs_dict.values()))
    
    conditions = kwargs["conditions"] if "conditions" in kwargs else all_epochs.metadata["trial_type"].unique()
    ch_types = kwargs["ch_types"] if "ch_types" in kwargs else ["hbo", "hbr"]
    cond_names = kwargs.get("cond_names", conditions)

    evoked_dict = {}
    for i, condition in enumerate(conditions):
        cname = cond_names[i]
        for ch_type in ch_types:
            evoked_dict[f"{cname}/{ch_type}"] = all_epochs[condition].average(
                picks=ch_type
            )
            evoked_dict[f"{cname}/{ch_type}"].rename_channels(lambda x: x[:-4])

    styles_dict = {}
    if 'hbr' in ch_types:
        styles_dict["hbr"] = dict(linestyle="dashed")

    fig = mne.viz.plot_compare_evokeds(
        evoked_dict,
        combine="mean",
        show_sensors="upper right",
        styles=styles_dict,
    )
    return fig


def plot_topo(epochs_dict, **kwargs):
    topomap_args = {
        "extrapolate": "local",
        "image_interp": 'linear',
        "time_format": "%d s",
        # 'contours': contours,
        "cmap": "RdBu_r",
        # 'colorbar': False,
        "cbar_fmt": "% 2.2f",
        "sensors": True,
    }
    if "vlim" in kwargs:
        topomap_args["vlim"] = kwargs["vlim"]
    if "times" in kwargs:
        topomap_args["times"] = kwargs["times"]

    all_epochs = mne.concatenate_epochs(list(epochs_dict.values()))
    if "conditions" not in kwargs:
        kwargs["conditions"] = all_epochs.metadata["trial_type"].unique()
    if "ch_types" not in kwargs:
        kwargs["ch_types"] = ["hbo", "hbr"]

    cond_names = kwargs.get("cond_names", kwargs["conditions"])

    for i, condition in enumerate(kwargs["conditions"]):
        cname = cond_names[i]
        for ch_type in kwargs["ch_types"]:
            evo = (
                all_epochs[condition]
                .average(picks=ch_type)
                .rename_channels(lambda x: x[:-4])
            )
            fig = evo.plot_topomap(
                **topomap_args,
                # title=f'{condition}/{ch_type}',
                show=False,
            )
            fig.suptitle(f"{cname}/{ch_type}")
            if "save_path" in kwargs:
                fig.savefig(Path(kwargs["save_path"]) / f"topo_{cname}_{ch_type}.png")
            plt.show()


def plot_joint(epochs_dict, **kwargs):
    topomap_args = {
        "extrapolate": "local",
        "image_interp": 'linear',
        "time_format": "%d s",
        # 'contours': contours,
        "cmap": "RdBu_r",
        # 'colorbar': False,
        "cbar_fmt": "% 2.2f",
        "sensors": True,
    }
    ts_args = {
        # 'ylim': dict(hbo=[vmin, vmax]),
        "spatial_colors": True,
    }
    if "vlim" in kwargs:
        topomap_args["vlim"] = kwargs["vlim"]
        ts_args["ylim"] = dict(hbo=kwargs["vlim"], hbr=kwargs["vlim"])

    all_epochs = mne.concatenate_epochs(list(epochs_dict.values()))
    if "conditions" not in kwargs:
        conditions = all_epochs.metadata["trial_type"].unique()
    else:
        conditions = kwargs["conditions"]

    cond_names = kwargs.get("cond_names", conditions)

    if "ch_types" not in kwargs:
        kwargs["ch_types"] = ["hbo", "hbr"]

    for i, condition in enumerate(conditions):
        cname = cond_names[i]
        for ch_type in kwargs["ch_types"]:
            evo = (
                all_epochs[condition]
                .average(picks=ch_type)
                .rename_channels(lambda x: x[:-4])
            )
            fig = evo.plot_joint(
                times=kwargs["times"] if "times" in kwargs else "peaks",
                topomap_args=topomap_args,
                ts_args=ts_args,
                title=f'{cname}/{ch_type}',
                show=False,
            )
            # fig.suptitle(f"{cname}/{ch_type}")
            if "save_path" in kwargs:
                fig.savefig(Path(kwargs["save_path"]) / f"joint_{cname}_{ch_type}.png")
            plt.show()
            
