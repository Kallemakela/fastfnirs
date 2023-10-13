import mne


def plot_evoked(epochs_dict, conditions=None):
    all_epochs = mne.concatenate_epochs(list(epochs_dict.values()))
    if conditions is None:
        conditions = all_epochs.metadata["trial_type"].unique()
    ch_types = ["hbo", "hbr"]

    evoked_dict = {}
    for condition in conditions:
        for ch_type in ch_types:
            evoked_dict[f"{condition}/{ch_type}"] = all_epochs[condition].average(
                picks=ch_type
            )
            evoked_dict[f"{condition}/{ch_type}"].rename_channels(lambda x: x[:-4])

    styles_dict = dict(hbr=dict(linestyle="dashed"))

    fig = mne.viz.plot_compare_evokeds(
        evoked_dict,
        combine="mean",
        show_sensors="upper right",
        styles=styles_dict,
    )
    return fig
