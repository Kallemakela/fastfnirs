import numpy as np
from scipy.interpolate import SmoothBivariateSpline, griddata


def interpolate(d, method="cubic", dx=None):
    """Interpolates nans of a 2D grid

    Parameters
    ----------
    d : np.ndarray, shape (w, h)
            data to interpolate
    method : str
            interpolation method, one of 'spline', 'linear', 'nearest', 'cubic'
    dx : int
            spline degree, default 3 if w > 5 and h > 5, else min(w, h) - 2

    Returns
    -------
    interpolated : (w, h)
    """

    h, w = d.shape
    mask = np.isnan(d)
    y, x = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    x1 = x[~mask]
    y1 = y[~mask]
    valid_d = d[~mask]

    if method == "spline":
        if dx is None:
            if w > 5 and h > 5:
                dx = 3
            else:
                dx = min(w, h) - 2
        spline = SmoothBivariateSpline(x1, y1, valid_d, kx=dx, ky=dx)
        x2, y2 = x[mask], y[mask]
        interpolated_points = spline.ev(x2, y2)
        interpolated = d.copy()
        np.putmask(interpolated, mask, interpolated_points)

    elif method == "cubic" or method == "linear" or method == "nearest":
        # interpolated = griddata((x1, y1), valid_d, (x, y), method=method)
        interpolated = griddata((y1, x1), valid_d, (y, x), method=method)

    else:
        raise ValueError('method must be one of "spline", "cubic", "linear", "nearest"')

    return interpolated


def interpolate_epochs(epochs, **kwargs):
    """Interpolate epochs in-place
    epochs: (n_epochs, n_chs, w, h, T)
    """
    n_epochs, n_chs, w, h, T = epochs.shape
    interpolated_epochs = np.zeros_like(epochs)
    for ei in range(n_epochs):
        for ci in range(n_chs):
            for ti in range(T):
                interpolated = interpolate(epochs[ei, ci, :, :, ti], **kwargs)
                if np.any(np.isnan(interpolated)):
                    interpolated = interpolate(interpolated, method="nearest")
                interpolated_epochs[ei, ci, :, :, ti] = interpolated
    return interpolated_epochs
