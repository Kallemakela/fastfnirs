import numpy as np
from fastfnirs.dataset.interpolation import interpolate
from joblib import Parallel, delayed


def to_grid(D, ch_names, ch2grid, c=2, w=11, h=5, reverse_y=True, **kwargs):
    """
    D: (n_epochs, n_channels, n_samples)
    ch_names: channels names, same order as D
    ch2grid: dict, channel name -> (x, y) grid coordinates

    Returns
    -------
    Dgrid: (n_epochs, c, h, w, n_samples)
    """
    # Dgrid = np.zeros((D.shape[0], c, h, w, D.shape[2]))
    Dgrid = np.nan * np.ones((D.shape[0], c, h, w, D.shape[2]))
    for ch_ix in range(D.shape[1]):
        ch = ch_names[ch_ix]
        x_ix, y_ix = ch2grid[ch]
        y_ix = h - 1 - y_ix if reverse_y else y_ix
        ch_type_ix = 0 if ch.endswith("hbo") else 1
        Dgrid[:, ch_type_ix, y_ix, x_ix, :] = D[:, ch_ix]
    return Dgrid


def X2grid_new(
    X, ch_names, ch2grid, to_grid_params={"h": 5, "w": 11, "c": 1}, **kwargs
):
    """
    Uses the same ch2grid for all subjects
    """
    print("Converting to grid")

    def X2grid_subject(Xs, ch_names, ch2grid, **kwargs):
        Xs_new = to_grid(Xs, ch_names, ch2grid, **kwargs)
        n_epochs, n_chs, h, w, T = Xs_new.shape
        for ei in range(n_epochs):
            for ci in range(n_chs):
                for ti in range(T):
                    # zero_ch_mask = (Xs_new[ei,ci,:,:,ti] == 0)
                    # print(f'{zero_ch_mask.astype(int)}')
                    # if ei % 30 == 0 and ti == T // 2: plot_grid(Xs_new[ei,ci,:,:,ti], title=f'Before interpolation')
                    # interpolated = interpolate(Xs_new[ei, ci, :, :, ti], **kwargs)
                    interpolated = interpolate(Xs_new[ei, ci, :, :, ti])
                    # zero_ch_mask = (interpolated == 0)
                    # print(f'{zero_ch_mask.astype(int)}')
                    # if ei % 30 == 0 and ti == T // 2: plot_grid(interpolated, title=f'After cubic interpolation')
                    if np.any(np.isnan(interpolated)):
                        interpolated = interpolate(interpolated, method="nearest")
                    # zero_ch_mask = (interpolated == 0)
                    # print(f'{zero_ch_mask.astype(int)}')
                    # if ei % 30 == 0 and ti == T // 2: plot_grid(interpolated, title=f'After corner interpolation')
                    Xs_new[ei, ci, :, :, ti] = interpolated
        return Xs_new

    X_new = Parallel(n_jobs=-1)(
        delayed(X2grid_subject)(X[sub], ch_names, ch2grid, **kwargs) for sub in X
    )
    # print('WARNING: not using parallel')
    # X_new = [X2grid_subject(X[sub], ch_names, ch2grid, **kwargs) for sub in X]
    X_new = {sub: X_new[i] for i, sub in enumerate(X)}
    return X_new


def bd2grid_ind(bd, to_grid_params={"h": 5, "w": 11, "c": 1}, **kwargs):
    """
    Uses individual ch2grid for each subject
    """

    print("Converting to grid")

    def X2grid_subject(Xs, chs, **kwargs):
        ch_names = [ch["ch_name"] for ch in chs]
        ch2grid = get_ch2grid(chs)
        Xs_new = to_grid(Xs, ch_names, ch2grid, **to_grid_params)
        n_epochs, n_chs, h, w, T = Xs_new.shape
        for ei in range(n_epochs):
            for ci in range(n_chs):
                for ti in range(T):
                    # zero_ch_mask = (Xs_new[ei,ci,:,:,ti] == 0)
                    # print(f'{zero_ch_mask.astype(int)}')
                    # if ei % 30 == 0 and ti == T // 2: plot_grid(Xs_new[ei,ci,:,:,ti], title=f'Before interpolation')
                    interpolated = interpolate(Xs_new[ei, ci, :, :, ti], **kwargs)
                    # zero_ch_mask = (interpolated == 0)
                    # print(f'{zero_ch_mask.astype(int)}')
                    # if ei % 30 == 0 and ti == T // 2: plot_grid(interpolated, title=f'After cubic interpolation')
                    if np.any(np.isnan(interpolated)):
                        interpolated = interpolate(interpolated, method="nearest")
                    # zero_ch_mask = (interpolated == 0)
                    # print(f'{zero_ch_mask.astype(int)}')
                    # if ei % 30 == 0 and ti == T // 2: plot_grid(interpolated, title=f'After corner interpolation')
                    Xs_new[ei, ci, :, :, ti] = interpolated
        return Xs_new

    # print('WARNING: not using parallel')
    # X_new = [
    # 	X2grid_subject(
    # 		bd.X[sub],
    # 		bd.epochs_dict[sub].info['chs'],
    # 		**kwargs,
    # 	)
    # 	for sub in bd.X
    # ]
    X_new = Parallel(n_jobs=-1)(
        delayed(X2grid_subject)(
            bd.X[sub],
            bd.epochs_dict[sub].info["chs"],
            **kwargs,
        )
        for sub in bd.X
    )
    X_new = {sub: X_new[i] for i, sub in enumerate(bd.X)}
    return X_new


def get_ch2grid(chs, w=11, h=5, reverse_y=True, **kwargs):
    """
    chs: mne.info['chs']
    """
    ch2grid = {}
    x_coords = [ch["loc"][0] for ch in chs]
    y_coords = [ch["loc"][1] for ch in chs]
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    x_step = (x_max - x_min) / w
    y_step = (y_max - y_min) / h
    for ch in chs:
        x, y, _ = ch["loc"][:3]
        x_ix = int((x - x_min - 1e-9) / x_step)
        y_ix = int((y - y_min - 1e-9) / y_step)
        if reverse_y:
            y_ix = h - y_ix - 1
        assert x_ix >= 0 and x_ix < w
        assert y_ix >= 0 and y_ix < h
        ch_name = ch["ch_name"]
        ch2grid[ch_name] = (x_ix, y_ix)
    return ch2grid


# %%
# bd = BrainDataset.load_data_2d(dataset)
# # bd = BrainDataset.load_data_grid(dataset)
# from dlnemo.utils import get_all_subjects
# train_subjects = get_all_subjects(dataset)[:-5]
# test_subjects = get_all_subjects(dataset)[-5:]
# d_train, d_test, info = bd.get_split_dataset(train_subjects, test_subjects)
# train_loader = torch.utils.data.DataLoader(d_train, batch_size=32, shuffle=True, num_workers=4)
# test_loader = torch.utils.data.DataLoader(d_test, batch_size=32, shuffle=False, num_workers=4)
# # %%
