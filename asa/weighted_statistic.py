from functools import wraps

import numpy as np
from sklearn.neighbors import KernelDensity

from .Bcorner import quantile as _quantile

# TODO: in batch by axis

def xw_asarray_wrapper(func):
    @wraps(func)
    def wrapper(x, w=None, **kwargs):
        w = np.asarray(w) if w is not None else None
        x = np.asarray(x)
        return func(x, w=w, **kwargs)
    return wrapper

def quantile(x, w=None, q=0.5, **kwargs):
    return np.array(_quantile(x, q, w, **kwargs)[0])


def median(x, w=None):
    return np.median(x) if w is None else quantile(x, w=w, q=0.5)

@xw_asarray_wrapper
def mean(x, w=None, axis=None, keepdims=False):
    if w is None:
        return np.mean(x, axis=axis, keepdims=keepdims)
    else:
        x, w = broadcast_arrays(x, w)
        return np.average(x, weights=w, axis=axis, keepdims=keepdims)

@xw_asarray_wrapper
def std(x,
        w=None,
        ddof=0,
        axis=None):  # sourcery skip: remove-unnecessary-else
    '''
    Note that the w here is considered as reliability weights.
    However, the weights in DescrStatsW is frequency weights.

    https://en.wikipedia.org/wiki/Weighted_arithmetic_mean#Weighted_sample_variance
    
    Because the ddof with weight is a so complex concept
    We use the effect sample size for ddof correction in weighted std
    https://en.wikipedia.org/wiki/Design_effect#Effective_sample_size
    '''

    if w is None:
        return np.std(x, ddof=ddof, axis=axis)
    else:
        x, w = broadcast_arrays(x, w)

        N = get_effect_sample_size(w, axis=axis) if w is not None else x.size

        _var = np.average((x - mean(x, w, axis=axis, keepdims=True))**2,
                          weights=w,
                          axis=axis) * N / (N - ddof)
        _var = np.atleast_1d(_var)
        _var[(N - ddof) <= 0] = np.nan
        return np.squeeze(np.sqrt(_var))


def std_mean(x, w=None, ddof=0):
    '''
    The ddof only affects the estimation of std
    We use the effect sample size as the denominator
    https://en.wikipedia.org/wiki/Design_effect#Effective_sample_size
    '''
    N = get_effect_sample_size(w) if w is not None else x.size
    return std(x, w=w, ddof=ddof) / np.sqrt(N)


def std_median(x, w=None, **kde_kwargs):
    kde = KernelDensity(**kde_kwargs).fit(x.reshape(-1, 1), sample_weight=w)
    m = median(x, w)
    if np.isnan(m):
        return np.nan
    fm = np.exp(kde.score_samples(np.array([[m]])))
    N = get_effect_sample_size(w) if w is not None else x.size
    return 1 / (2 * fm * np.sqrt(N))[0]


def std_std(x, w=None, ddof=0):
    '''
    The ddof only affects the estimation of std
    We use the effect sample size as the denominator
    '''
    N = get_effect_sample_size(w) if w is not None else x.size
    factor = np.sqrt((N**2 - 1) / (N - ddof)**2 - 1)
    return std(x, w=w, ddof=ddof) * factor / 2


def get_effect_sample_size(w, axis=None):
    V1 = np.sum(w, axis=axis)
    V2 = np.sum(w**2, axis=axis)
    return V1**2 / V2


def broadcast_arrays(*args):
    """
    Broadcasts any number of arrays against each other.

    Parameters:
    - args: array-like - The arrays to broadcast.

    Returns:
    - broadcasted_arrays: list - The broadcasted arrays.

    Notes:
    Different `np.broadcast_arrays`, which do not support broadcast (n, ) to (n, m)
    """
    if len(args) == 0:
        return []

    ones_lst = [np.ones_like(arg) for arg in args]
    all_ones = ones_lst[0]
    for ones in ones_lst[1:]:
        all_ones = all_ones * ones

    broadcasted_arrays = [arg * all_ones for arg in args]
    return broadcasted_arrays
