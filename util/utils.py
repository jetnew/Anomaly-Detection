import numpy as np
import pandas as pd

def get_window(series, backward=5, forward=0, slide=1, pad=False):
    """
    Get sliding windows given a list.

    Params:
        backward - No. of timestamps backward to include in sliding window
        forward - No. of timestamps forward to include in sliding window
        slide - No. of timestamps to skip for each window
        pad - Bool: If true, get sliding windows for every point in the series. Ends of the array are padded with np.nan.

    E.g.
        backward = 3
        forward = 3
        slide = 1

        arr = [1,2,3,4,5,6,7,8,9]
        sliding_windows = get_window(arr, backward=backward, forward=forward, slide=slide)
        sliding_windows = [[1,2,3,4,5,6,7],
                           [2,3,4,5,6,7,8],
                           [3,4,5,6,7,8,9]]
    """
    series = list(series)
    s_len = len(series)
    sliding_window = []

    if pad:
        for i in range(backward):
            window = [np.nan for i in range(backward + 1 + forward)]
            sliding_window.append(window)

    for i in range(backward, s_len - forward, slide):
        window = series[i - backward:i + forward + 1]
        sliding_window.append(window)

    if pad:
        for i in range(forward):
            window = [np.nan for i in range(backward + 1 + forward)]
            sliding_window.append(window)

    return np.array(sliding_window)

