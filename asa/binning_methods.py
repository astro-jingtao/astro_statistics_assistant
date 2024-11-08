from functools import partial

import numpy as np
from scipy.stats import binned_statistic, binned_statistic_2d
from astropy.stats import poisson_conf_interval

from . import weighted_statistic as w
from .utils import flag_bad

_range = range


def binned_statistic_robust(x, values, statistic='mean', bins=10, range=None):
    '''
    Deal with the nan and inf:
        statistic_res: ignore
        bin_edges: ignore
        binnumber: -1
    '''
    is_bad = flag_bad(x) | flag_bad(values)
    _x = x[~is_bad]
    _values = values[~is_bad]

    statistic_res, bin_edges, _binnumber = binned_statistic(
        _x, _values, statistic=statistic, bins=bins, range=range)

    binnumber = np.full(len(x), -1)
    binnumber[~is_bad] = _binnumber

    return statistic_res, bin_edges, binnumber


def binned_statistic_2d_robust(x,
                               y,
                               values,
                               statistic='mean',
                               bins=10,
                               range=None):
    '''
    Note that expand_binnumbers is always True
    '''

    is_bad = flag_bad(x) | flag_bad(y) | flag_bad(values)
    _x = x[~is_bad]
    _y = y[~is_bad]
    _values = values[~is_bad]

    statistic_res, x_edge, y_edge, _binnumber = binned_statistic_2d(
        _x,
        _y,
        _values,
        statistic=statistic,
        bins=bins,
        range=range,
        expand_binnumbers=True)

    binnumber_x = np.full(len(x), -1)
    binnumber_y = np.full(len(y), -1)
    binnumber_x[~is_bad] = _binnumber[0]
    binnumber_y[~is_bad] = _binnumber[1]
    binnumber = np.stack([binnumber_x, binnumber_y], axis=0)

    return statistic_res, x_edge, y_edge, binnumber


def weighted_binned_statistic(x, y, w, bins=10, statistic=None, range=None):
    # TODO: least number in each bin
    _, edges, bin_index = binned_statistic(x,
                                           y,
                                           statistic='count',
                                           bins=bins,
                                           range=range)

    return np.array([
        statistic(y[bin_index == i], w[bin_index == i])
        for i in _range(1, len(edges))
    ])


def bin_2d(x, y, z, bins=10, range=None, min_data=0):
    # TODO: support with weights, support more statistic
    Z, x_edges, y_edges, _ = binned_statistic_2d(x,
                                                 y,
                                                 z,
                                                 statistic='mean',
                                                 bins=bins,
                                                 range=range)
    N, x_edges, y_edges, _ = binned_statistic_2d(x,
                                                 y,
                                                 z,
                                                 statistic='count',
                                                 bins=bins,
                                                 range=range)
    Z[N <= min_data] = np.nan
    Z = Z.T
    x_center, y_center = 0.5 * (x_edges[1:] + x_edges[:-1]), 0.5 * (
        y_edges[1:] + y_edges[:-1])
    X, Y = np.meshgrid(x_center, y_center)
    return X, Y, Z, x_edges, y_edges


def bin_1d(x,
           y,
           weights=None,
           x_statistic=None,
           y_statistic=None,
           bins=10,
           range=None,
           min_data=0):
    # TODO: count, std_std
    '''
    input:
        x_statistic, List[str]:
            'mean', 'median', 'std', 'std_mean', 'std_median', 'std_std', 'q:x' (x is a number between 0 and 1)
        y_statistic, List[str]:
            'mean', 'median', 'std', 'std_mean', 'std_median', 'std_std', 'q:x' (x is a number between 0 and 1)
    
    notes:
        ...
    '''

    if x_statistic is None:
        x_statistic = []
    elif isinstance(x_statistic, str):
        x_statistic = [x_statistic]

    if y_statistic is None:
        y_statistic = ['mean']
    elif isinstance(y_statistic, str):
        y_statistic = [y_statistic]

    _, edges, bin_index = binned_statistic_robust(x,
                                                  y,
                                                  statistic='count',
                                                  bins=bins,
                                                  range=range)

    # sourcery skip: dict-comprehension
    statistic = {}
    for x_stat in x_statistic:
        statistic[f'x_{x_stat}'] = []
    for y_stat in y_statistic:
        statistic[f'y_{y_stat}'] = []

    for i in _range(1, len(edges)):
        in_this_bin = (bin_index == i)
        N_in_this_bin = in_this_bin.sum()

        for x_stat in x_statistic:
            if N_in_this_bin <= min_data:
                statistic[f'x_{x_stat}'].append(np.nan)
            else:
                _weights = None if weights is None else weights[in_this_bin]
                statistic[f'x_{x_stat}'].append(
                    get_stat_method(x_stat)(x[in_this_bin], _weights))
        for y_stat in y_statistic:
            if N_in_this_bin <= min_data:
                statistic[f'y_{y_stat}'].append(np.nan)
            else:
                _weights = None if weights is None else weights[in_this_bin]
                statistic[f'y_{y_stat}'].append(
                    get_stat_method(y_stat)(y[in_this_bin], _weights))

    center = 0.5 * (edges[1:] + edges[:-1])

    # to np.ndarray
    for k in statistic:
        statistic[k] = np.asarray(statistic[k])

    return center, edges, bin_index, statistic


def get_stat_method(stat_name):
    mapper = {
        'mean': w.mean,
        'median': w.median,
        'std': partial(w.std, ddof=1),
        'std_mean': partial(w.std_mean, ddof=1),
        'std_median': partial(w.std_median, bandwidth='silverman'),
        'std_std': partial(w.std_std, ddof=1),
    }
    if stat_name.startswith('q:'):
        return partial(w.quantile, q=float(stat_name[2:]))
    else:
        return mapper[stat_name]


def get_epdf(x,
             bins=10,
             range=None,
             weights=None,
             density=False,
             interval="frequentist-confidence",
             sigma=1,
             background=0,
             confidence_level=None):
    """
    Estimate the empirical probability density function (EPDF) of a dataset.

    Parameters
    ----------
    x : array_like
        Input array of data points.
    bins : int or sequence of scalars or str, optional
        If an int, it defines the number of equal-width bins in the range.
        If a sequence, it defines the bin edges, including the rightmost edge.
        Default is 10.
    range : (float, float), optional
        The lower and upper range of the bins. If not provided, range is simply
        (x.min(), x.max()). Values outside the range are ignored.
    weights : array_like, optional
        An array of weights, of the same shape as `x`. Each value in `x` only contributes its associated weight towards the bin count
        (instead of 1).
    density : bool, optional
        If False, the result contains the number of samples in each bin. If True, the result is the value of the probability
        density function at the bin, normalized such that the integral over the range is 1.
        Default is False.
    interval : str, optional
        The type of interval to calculate. 
        Used by astropy.stats.poisson_conf_interval
        Default is 'frequentist-confidence'.
    sigma : float, optional
        Used by astropy.stats.poisson_conf_interval to calculate the interval.
        Default is 1.
    background : float, optional
        Used by astropy.stats.poisson_conf_interval to calculate the interval.
        Default is 0.
    confidence_level : float, optional
        Used by astropy.stats.poisson_conf_interval to calculate the interval.
        Default is None.

    Returns
    -------
    centers : ndarray
        The bin centers.
    N : ndarray
        The values of the histogram.
    lower : ndarray
        The lower bounds of the interval.
    upper : ndarray
        The upper bounds of the interval.
    edges : ndarray
        The bin edges.
    d_bin : float
        The width of each bin.

    Notes
    -----
    The function uses NumPy's `np.histogram` to compute the histogram and `astropy.stats.poisson_conf_interval` to calculate the confidence
    interval for the Poisson distribution. 

    Examples
    --------
    >>> import numpy as np
    >>> x = np.random.normal(0, 1, 1000)
    >>> centers, N, lower, upper, edges, d_bin = get_epdf(x, bins=20, density=True)
    """

    N, edges = np.histogram(x,
                            bins=bins,
                            range=range,
                            weights=weights,
                            density=False)
    lower, upper = poisson_conf_interval(N,
                                         interval=interval,
                                         sigma=sigma,
                                         background=background,
                                         confidence_level=confidence_level)
    centers = (edges[:-1] + edges[1:]) / 2

    d_bin = edges[1:] - edges[:-1]

    if density:
        scaler = np.sum(N) * d_bin
        N = N / scaler
        lower = lower / scaler
        upper = upper / scaler
    
    return centers, N, lower, upper, edges, d_bin

