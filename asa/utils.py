import numpy as np
from scipy.stats import binned_statistic

_range = range

def flag_bad(x):
    """
    It returns True if the input is NaN or Inf, and False otherwise
    
    :param x: the input data
    :return: A boolean array of the same shape as x, where True indicates that the corresponding element
    of x is NaN or +/-inf.
    """
    return np.isnan(x) | np.isinf(x)


def string_to_list(string):
    return [string] if isinstance(string, str) else string


def is_string_or_list_of_string(x):
    return (isinstance(x, str)
            or isinstance(x, list) and all(isinstance(y, str) for y in x))


def weighted_binned_statistic(x, y, w, bins=10, statistic=None, range=None):
    _, edges, bin_index = binned_statistic(x,
                                           y,
                                           statistic='count',
                                           bins=bins,
                                           range=range)

    return np.array([
        statistic(y[bin_index == i], w[bin_index == i])
        for i in _range(1, len(edges))
    ])
