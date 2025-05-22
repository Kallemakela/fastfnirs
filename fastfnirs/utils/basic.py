def split_list(lst, n_parts):
    """Split a list into n_parts as evenly as possible."""
    k, m = divmod(len(lst), n_parts)
    return [
        lst[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n_parts)
    ]


def reverse_dict(d):
    return {v: k for k, v in d.items()}


def centered_moving_average(scores, window_size=11):
    assert window_size % 2 == 1, "Window size must be odd for centered MA"
    pad = window_size // 2
    padded = np.pad(scores, (pad, pad), mode="edge")
    return np.convolve(padded, np.ones(window_size) / window_size, mode="valid")
