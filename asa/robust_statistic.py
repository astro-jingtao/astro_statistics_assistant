import numpy as np
from . import weighted_statistic as ws


def sigma_clip(x,
               n_sigma=3,
               n=10,
               x_err=None,
               use_median=True,
               use_quantile=True):
    """
    Perform sigma-clipping on a dataset to remove outliers.

    Parameters
    ----------
    x : array_like
        The input data array to be clipped.
    n_sigma : float, optional
        The number of standard deviations to use for clipping. Default is 3.
    n : int, optional
        The maximum number of iterations to perform. Default is 10.
    x_err : array_like, optional
        The uncertainties in the input data.
    use_median: bool, optional
        If True, the median is used as the central estimator. If False, the
        mean is used. Default is True.
    use_quantile: bool, optional
        If True, the standard deviation is calculated as the half of the difference
        between the 84th and 16th percentiles. If False, the standard deviation
        is calculated in the usual way. Default is True.

    Returns
    -------
    m : float
        The clipped mean of the data.
    s : float
        The clipped standard deviation of the data.
    is_good : array_like of bool
        A boolean array indicating which data points are considered good (i.e.,
        not outliers) after clipping.

    Notes
    -----
    This function performs sigma-clipping on the input data `x` by iteratively
    calculating the mean and standard deviation of the data, excluding points
    that are more than `n_sigma` standard deviations away from the mean. The
    process is repeated for a maximum of `n` iterations or until convergence
    (i.e., when the set of good points does not change between iterations).

    If uncertainties `x_err` are provided, they are used to weight the data
    points in the calculation of the mean and standard deviation. If not
    provided, all points are treated as having equal weight.

    If uncertainties `x_err` are provided, the standard deviation of the each data
    point is calculated as the quadrature sum of the intrinsic standard deviation
    and the uncertainty in the data point.

    The function returns the clipped mean `m`, the clipped standard deviation
    `s`, and a boolean array `is_good` that indicates which data points are
    considered good after the clipping process.

    Examples
    --------
    >>> data = np.array([1, 2, 3, 4, 100])
    >>> m, s, is_good = sigma_clip(data)
    >>> m
    2.5
    >>> s
    0.0
    >>> is_good
    array([ True,  True,  True,  True, False])
    """

    if x_err is not None:
        w = 1 / x_err**2
    else:
        x_err = np.zeros_like(x)
        w = np.ones_like(x)

    cen_func = ws.median if use_median else ws.mean

    if use_quantile:

        def std_func(x, w):
            return (ws.quantile(x, w, q=0.84) - ws.quantile(x, w, q=0.16)) / 2
    else:
        std_func = ws.std

    is_good = np.ones_like(x, dtype=bool)
    m_old = np.nan
    s_old = np.nan

    for _ in range(n):
        m = cen_func(x[is_good], w[is_good])
        s = std_func(x[is_good], w[is_good])
        sigma_in = np.sqrt(s**2 + x_err**2)
        is_good = np.abs(x - m) < n_sigma * sigma_in
        if (m == m_old) and (s == s_old):
            break
        m_old = m
        s_old = s

    return m, s, is_good
