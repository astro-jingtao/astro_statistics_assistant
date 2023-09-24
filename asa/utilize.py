import numpy as np


def flag_bad(x):
    """
    It returns True if the input is NaN or Inf, and False otherwise
    
    :param x: the input data
    :return: A boolean array of the same shape as x, where True indicates that the corresponding element
    of x is NaN or +/-inf.
    """
    return np.isnan(x) | np.isinf(x)