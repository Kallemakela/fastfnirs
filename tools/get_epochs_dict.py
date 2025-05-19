def get_epochs_dict(dataset, include_events=None, task="b_pn", **kwargs):
    info = {}
    if dataset == "nemo":
        from fastfnirs.epochs import get_epochs
        from fastfnirs.utils import get_event_name_mapping_for_task

        include_events = include_events or "empe"
        use_3d_montage = kwargs.pop("use_3d_montage", False)
        if kwargs.get("subjects") == "all" or kwargs.get("subjects") == ["all"]:
            kwargs["subjects"] = get_all_subjects()
        epochs_dict = get_epochs(include_events=include_events, **kwargs)
        event_name_mapping = get_event_name_mapping_for_task(task, include_events)
        if use_3d_montage:
            mtg_path = get_cwd() / "processed_data/montage/nemo_montage_3d.pkl"
            nemo_3d_montage = load_from(mtg_path)
            print(f"Using 3D montage from {mtg_path}")
            for subject in epochs_dict:
                epochs_dict[subject].set_montage(nemo_3d_montage)
    elif dataset == "mima":
        from fastfnirs.load_external_dataset import load_epochs_mima_mne

        epochs_dict, event_name_mapping, info = load_epochs_mima_mne(**kwargs)
    elif dataset == "bnci":
        bnci_data_path = Path(config["bnci_data_path"])
        epochs_dict = load_from(bnci_data_path / "epochs_dict.pkl")
        event_name_mapping = load_from(bnci_data_path / "name2event.pkl")
    elif dataset == "mnea":
        mnea_data_path = Path(config["mnea_data_path"])
        epochs_dict = load_from(mnea_data_path)
        task = "fingervsfoot"
        event_name_mapping = load_from(
            mnea_data_path.parent / f"name2eventid_{task}.pkl"
        )

        # Remove channels below threshold
        epochs_dict = remove_chs_below_threshold(epochs_dict, threshold=-0.04)
    elif dataset == "offt":
        offt_data_path = Path(config["offt_data_path"])
        epochs_dict = load_from(offt_data_path)
        task = "rightvsfoot"  # right vs foot
        # task = 'fingervsfoot' # left and right vs foot
        # task = '3class'
        event_name_mapping = load_from(
            offt_data_path.parent / f"name2eventid_{task}.pkl"
        )
        epochs_dict = {
            k: v[list(event_name_mapping.keys())] for k, v in epochs_dict.items()
        }
    elif dataset == "fpar":
        fpar_data_path = Path(config["fpar_data_path"])
        epochs_dict = load_from(fpar_data_path)
        event_name_mapping = load_from(fpar_data_path.parent / "name2eventid.pkl")
        epochs_dict = {
            k: v[list(event_name_mapping.keys())] for k, v in epochs_dict.items()
        }
    elif dataset == "emob":
        epochs_dict, event_name_mapping = load_from(
            get_cwd() / f"data/{dataset}/epochs_dict.pkl"
        ), load_from(get_cwd() / f"data/{dataset}/name2event.pkl")
    elif dataset == "evsr":  # NEMO EMPE vs AFIM
        empe_epochs_dict, empe_event_name_mapping, empe_info = get_epochs_dict(
            "nemo", include_events="empe", task="4_class", **kwargs
        )
        afim_epochs_dict, afim_event_name_mapping, afim_info = get_epochs_dict(
            "nemo", include_events="afim", task="4_class", **kwargs
        )
        comb_epochs_dict = {}
        for s in empe_epochs_dict:
            # add 4 to afim events
            afim_epochs_dict[s].event_id = {
                k: v + 4 for k, v in afim_epochs_dict[s].event_id.items()
            }
            afim_epochs_dict[s].events[:, 2] += 4
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    category=RuntimeWarning,
                    message="Concatenation of Annotations within Epochs is not supported yet. All annotations will be dropped.",
                )
                comb_epochs_dict[s] = mne.concatenate_epochs(
                    [empe_epochs_dict[s], afim_epochs_dict[s]], verbose=False
                )
            # update epoch
            comb_epochs_dict[s].metadata["epoch"] = np.arange(len(comb_epochs_dict[s]))
        comb_event_name_mapping = {**empe_event_name_mapping, **afim_event_name_mapping}
        comb_event_name_mapping = {
            k: 0 if "afim_" in k else 1 for k, v in comb_event_name_mapping.items()
        }
        comb_info = {**empe_info, **afim_info}
        epochs_dict, event_name_mapping, info = (
            comb_epochs_dict,
            comb_event_name_mapping,
            comb_info,
        )
    else:
        raise ValueError(f"Unknown dataset {dataset}")
    return epochs_dict, event_name_mapping, info
