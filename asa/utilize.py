import numpy as np


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
    return (
        isinstance(x, str)
        or isinstance(x, list)
        and all(isinstance(y, str) for y in x)
    )