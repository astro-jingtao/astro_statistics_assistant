import sys
from typing import List, Dict

import numpy as np

_range = range


def flag_bad(x):
    """
    It returns True if the input is NaN or Inf, and False otherwise

    :param x: the input data
    :return: A boolean array of the same shape as x, where True indicates that the corresponding element
    of x is NaN or +/-inf.
    """
    return np.isnan(x) | np.isinf(x)


def balance_class(x, y, random_state=None):
    """
    Balance the classes in a dataset by randomly removing data points from the majority class(es).

    :param x: the input data
    :param y: the labels
    :param random_state: the random state to use for the random number generator
    :return: the balanced data and labels
    """
    if random_state is not None:
        np.random.seed(random_state)
    unique, counts = np.unique(y, return_counts=True)
    min_count = np.min(counts)
    idx = np.concatenate([
        np.random.choice(np.where(y == u)[0], min_count, replace=False)
        for u in unique
    ])
    return x[idx], y[idx]


def is_float(x):
    '''
    check if x is kind of float, such as built-in float, np.float32, np.float64, np.ndarray of float, list of float...
    '''
    return (isinstance(x, (float, np.float32, np.float64))
            or (isinstance(x, np.ndarray) and x.dtype.kind == 'f')
            or (isinstance(x, list) and all(isinstance(y, float) for y in x)))


def is_int(x):
    '''
    check if x is kind of int, such as built-in int, np.int32, np.int64, np.ndarray of int, list of int...
    '''
    return (isinstance(x, (int, np.int32, np.int64))
            or (isinstance(x, np.ndarray) and x.dtype.kind == 'i')
            or (isinstance(x, list) and all(isinstance(y, int) for y in x)))


def is_bool(x):
    '''
    check if x is kind of bool, such as built-in bool, np.bool, np.ndarray of bool, list of bool...
    '''
    return isinstance(
        x, bool) or (isinstance(x, np.ndarray) and x.dtype.kind == 'b') or (
            isinstance(x, list) and all(isinstance(y, bool) for y in x))


def string_to_list(string):
    return [string] if isinstance(string, str) else string


def is_string_or_list_of_string(x):
    return (isinstance(x, str)
            or isinstance(x, list) and all(isinstance(y, str) for y in x))


def list_reshape(lst: List, shape) -> List[List]:
    """
    Reshape a list into a list of lists
    :param lst: the input list
    :param shape: the shape of the output list
    :return: a list of lists
    """
    return [lst[i:i + shape[1]] for i in _range(0, len(lst), shape[1])]


def auto_set_range(x, y, _range, auto_p):
    if _range is None:
        _range = [[x.min(), x.max()], [y.min(), y.max()]]
    elif _range == 'auto':
        if auto_p is None:
            auto_p = ([1, 99], [1, 99])
        _range = [[
            np.percentile(x, auto_p[0][0]),
            np.percentile(x, auto_p[0][1])
        ], [np.percentile(y, auto_p[1][0]),
            np.percentile(y, auto_p[1][1])]]
    return _range


def get_kwargs_each(fixed_kwargs, changed_kwargs, shape):
    """
    Get a list of kwargs for each element in a 2D array

    :param fixed_kwargs: a dictionary of fixed kwargs
    :param changed_kwargs: a dictionary of changed kwargs
    :param shape: the shape of the 2D array
    :return: a list of kwargs
    """
    kwargs_each = []
    for i in _range(shape[0]):
        kwargs_each.append([])
        for j in _range(shape[1]):
            kwargs_each[-1].append({})
            for key, value in fixed_kwargs.items():
                kwargs_each[-1][-1][key] = value
            for key, value in changed_kwargs.items():
                kwargs_each[-1][-1][key] = value[i][j]
    return kwargs_each


def remove_bad(xs):
    if xs[0].ndim == 1:
        n_sample = len(xs[0])
    else:
        n_sample = xs[0].shape[0]
    bad = np.zeros(n_sample, dtype=bool)
    for x in xs:
        if x.ndim == 1:
            bad |= flag_bad(x)
        else:
            bad |= flag_bad(x).any(axis=1)
    return [x[~bad] for x in xs]


def all_asarray(xs):
    return [np.asarray(x) for x in xs]


def is_empty(x):
    return len(x) == 0



def get_rank(x):
    return np.argsort(np.argsort(x))


def to_little_endian(x):
    """
    It takes a numpy array and returns a new numpy array with little-endian format.
    It checks if the array is already in little-endian format and returns it directly if so.

    :param x: A numpy array.
    :return: A numpy array in little-endian byte order.
    """
    x = np.array(x)
    if (x.dtype.byteorder == '<') or (x.dtype.byteorder == '='
                                      and sys.byteorder == 'little'):
        # If the array is already little-endian, return it as is.
        return x
    else:
        # Otherwise, convert to little-endian.
        return x.byteswap().newbyteorder('little')


def xy2ij_imshow(x, y, img_shape, extent, origin):
    """
    Convert x, y coordinates to the index of the image.

    Parameters:
    - x: array-like - The x-coordinate of the points.
    - y: array-like - The y-coordinate of the points.
    - img_shape: array-like - The shape of the image.
    - extent: array-like - The extent of the image.
    - origin: str - The origin of the image.

    Returns:
    - i: array-like - The first index of the image.
    - j: array-like - The second index of the image.

    """
    x = np.asarray(x)
    y = np.asarray(y)

    dx = (extent[1] - extent[0]) / img_shape[1]
    dy = (extent[3] - extent[2]) / img_shape[0]

    i = np.floor((y - extent[2]) / dy)
    j = np.floor((x - extent[0]) / dx)

    if origin == 'upper':
        i = img_shape[0] - i

    return int(i), int(j)