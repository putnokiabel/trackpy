import numpy as np
from pandas import DataFrame
import warnings


def apply_spiff_correction(f: DataFrame, pos_columns=None) -> DataFrame:
    """
    Removes pixel bias in a given list of features (using a single-pixel interior filling function),
    thereby improving sub-pixel accuracy.

    Parameters
    ----------
    :param f: DataFrame
    :param pos_columns: list of column names, optional
    :return: DataFrame

    Notes
    ----------
    The algorithm used is inspired by "Analysis and correction of errors in
    nanoscale particle tracking using the Single-pixel interior filling function
    (SPIFF) algorithm" by Yuval et al.
    The accuracy of this algorithm improves with the number of features. When
    tracking features across multiple frames (e.g. in a video), consider locating
    the features across all frames first (using tp.batch) before applying this function
    (as opposed to applying this function for each individual frame).
    If f contains less than 100 features, f is returned as-is, due to lack of data.
    """
    if len(f) < 100:
        warnings.warn("Not enough features to apply pixel bias correction")
        return f

    if pos_columns is None:
        if 'z' in f:
            pos_columns = ['x', 'y', 'z']
        else:
            pos_columns = ['x', 'y']

    f = f.copy()

    # Correct each column individually (subject to further optimization,
    # by assuming sub-pixel bias is equal across certain dimensions).
    for col in pos_columns:
        # Get the values as a numpy array
        x = np.array(f[col])

        # Get the sub-pixel values
        spiff = x % 1

        # Mirror the values around the center of the pixel
        spiff_mirrored = np.where(spiff < 0.5, spiff, 1 - spiff)

        # Sort the values for efficient search
        spiff_sorted = np.sort(spiff_mirrored)

        # Reverse any sub-pixel bias
        spiff_corrected_low = np.searchsorted(spiff_sorted, spiff) / len(x) / 2
        spiff_corrected_high = 1 - np.searchsorted(spiff_sorted, 1 - spiff) / len(x) / 2
        spiff_corrected = np.where(spiff < 0.5, spiff_corrected_low, spiff_corrected_high)

        # Add the sub-pixel value back to the original pixel position
        x_corrected = np.floor(x) + spiff_corrected
        f[col] = x_corrected

    return f
