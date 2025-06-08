import warnings

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.scale import scale_factory
from scipy.stats import gaussian_kde
from statsmodels.stats.weightstats import DescrStatsW

from .Bcorner import corner, hist2d, quantile
from .binning_methods import bin_1d, bin_2d, get_epdf
from .loess2d import loess_2d_map
from .plot_utils import ColorCycler, jitter_data
from .utils import (all_asarray, any_empty, auto_set_range, flag_bad, is_empty,
                    remove_bad)
from .correlation_methods import get_correlation_coefficients

# TODO: extract common code
# - flag and remove bad

TREND_QUANTILE_ALIASES = ['q', 'quantile', 'percentile']
TREND_STD_ALIASES = ['s', 'sigma', 'std']
TREND_IQR_ALIASES = ['iqr', 'interquartile']
TREND_STD_MEAN_ALIASES = ['std_mean']
TREMD_STD_MEDIAN_ALIASES = ['std_median']

trend_color_cycle = ColorCycler()


def plot_trend(x,
               y,
               bins=20,
               quantile=False,
               x_method='center',
               y_method='median',
               yerr_method='quantile',
               yerr_args=None,
               is_x_interval=False,
               fbetween_method=None,
               fbetween_args=None,
               fbetween_bad_policy='omit',
               ax=None,
               range=None,
               auto_p=None,
               weights=None,
               N_min=1,
               color=None,
               errorbar_kwargs=None,
               fbetween_kwargs=None,
               plot_kwargs=None):
    # TODO: fix range and auto_p
    """
    Plot the trend line between two variables with options for error bars and shadowed intervals.

    Parameters
    ----------
    x : array_like, shape (nsamples,)
        The independent variable samples.

    y : array_like, shape (nsamples,)
        The dependent variable samples.

    bins : int, optional
        The number of bins to use for the histogram. Default is 20.

    x_method : str, optional
        Method to calculate statistics for x bins. Options are:
        'center' (default), 'mean', 'median'.

    y_method : str or None, optional
        Method to calculate statistics for y values. Options are:
        'mean', 'median'. Default is 'median'.

    yerr_method : str or None, optional
        Method for calculating the y error values. Default is 'quantile'.

    yerr_args : tuple or None, optional
        Arguments for the yerr_method. Default is None.

    is_x_interval: bool, optional
        If True, plot the interval between x values. Default is False.

    fbetween_method : str or None, optional
        Method to calculate filled area between values. Default is None.

    fbetween_args : tuple or None, optional
        Arguments for fbetween_method. Default is None.

    fbetween_bad_policy : str, optional
        Policy for handling bad values in fbetween_method. Default is 'omit'.
        'omit' will omit the bad values (default behavior for `fill_between` in matplotlib).
        'skip' will skip the bad values.
        'zero' will set the bad values to zero
        will be ignored if is_x_interval is True.

    ax : matplotlib.axes.Axes, optional
        The axes on which to plot. If None, uses the current axes.

    range : array_like, shape (2,), optional
        The range of x values to include in the plot. Default is None.

    auto_p : float or None, optional
        The quantile for the automatic range. Default is None.

    weights : array_like, shape (nsamples,), optional
        Sample weights. Default is None.

    N_min : int, optional
        Minimum number of samples in each bin to display the trend. Default is 1.

    color : str or None, optional
        The color for the trend line. If None, uses the next color from the cycle.

    errorbar_kwargs : dict, optional
        Additional keyword arguments for the error bar plotting.

    fbetween_kwargs : dict, optional
        Additional keyword arguments for the fill between plotting.

    plot_kwargs : dict, optional
        Additional keyword arguments for the line plotting.

    Returns
    -------
    None
        The function will display the plot but does not return any values.

    Notes
    -----
    - The function accepts various methods for statistical calculation which allows flexibility in visualization.
    - Ensure that the lengths of x, y, and weights are consistent.
    - Custom plotting parameters can be passed through the respective kwargs.
    """

    if ax is None:
        ax = plt.gca()

    if plot_kwargs is None:
        plot_kwargs = {}

    x, y, weights = remove_bad([x, y, weights])

    if is_empty([x, y, weights]):
        warnings.warn(
            "The x or y are empty after remove bad data, skip the plot")
        return

    # TODO: better auto_set_range
    if (range is None) or range == 'auto':
        range = auto_set_range(x, x, range, auto_p)[0]

    if weights is None:
        weights = np.ones_like(x)

    if color is None:
        color = trend_color_cycle.next(ax=ax)

    if x_method == 'center':
        x_statistic = None
    elif x_method == 'median':
        x_statistic = ['median']
    elif x_method == 'mean':
        x_statistic = ['mean']
    else:
        raise ValueError(f"x_type {x_method} is not supported")

    if y_method is None:
        if yerr_method in TREND_STD_ALIASES:
            raise ValueError(
                "yerr_method is set as sigma, but y_method is None")
        y_statistic = []
    elif y_method == 'median':
        y_statistic = ['median']
    elif y_method == 'mean':
        y_statistic = ['mean']
    else:
        raise ValueError(f"y_type {y_method} is not supported")

    y_statistic_err, yerr_args = _trend_get_y_statistic(yerr_method, yerr_args)
    y_statistic_fbetween, fbetween_args = _trend_get_y_statistic(
        fbetween_method, fbetween_args)

    y_statistic = y_statistic + y_statistic_err + y_statistic_fbetween

    if len(y_statistic) == 0:
        y_statistic = None
    else:
        y_statistic = list(set(y_statistic))

    x_center, x_edges, _, statistic = bin_1d(x,
                                             y,
                                             weights=weights,
                                             x_statistic=x_statistic,
                                             y_statistic=y_statistic,
                                             bins=bins,
                                             quantile=quantile,
                                             range=range,
                                             min_data=N_min)

    if x_method == 'center':
        x_bin = x_center
    else:
        x_bin = statistic[f'x_{x_method}']

    if y_method is None:
        y_bin = None
    else:
        y_bin = statistic[f'y_{y_method}']

    if is_x_interval:
        return _trend_interval(ax, x_edges, x_bin, y_bin, statistic, color,
                               plot_kwargs, yerr_method, yerr_args,
                               errorbar_kwargs, fbetween_method, fbetween_args,
                               fbetween_kwargs)

    if y_bin is not None:

        if plot_kwargs is None:
            plot_kwargs = {}

        if "color" not in plot_kwargs:
            plot_kwargs["color"] = color

        _x_bin, _y_bin = remove_bad([x_bin, y_bin])
        ax.plot(_x_bin, _y_bin, **plot_kwargs)

    if yerr_method is not None:

        yerr_low, yerr_up = _trend_get_lower_upper(y_bin, yerr_method,
                                                   yerr_args, statistic)
        if y_bin is None:
            _y_bin = (yerr_low + yerr_up) / 2  # a fake value
        else:
            _y_bin = y_bin

        yerr = (_y_bin - yerr_low, yerr_up - _y_bin)

        if errorbar_kwargs is None:
            errorbar_kwargs = {}

        if "color" not in errorbar_kwargs:
            errorbar_kwargs["color"] = color
        if "linestyle" not in errorbar_kwargs:
            errorbar_kwargs["linestyle"] = ""

        is_bad = flag_bad(x_bin) | flag_bad(_y_bin)
        _x_bin, _y_bin, _yerr = x_bin[~is_bad], _y_bin[~is_bad], np.asarray(
            yerr)[:, ~is_bad]
        ax.errorbar(_x_bin, _y_bin, yerr=_yerr, **errorbar_kwargs)

    if fbetween_method is not None:

        fbetween = _trend_get_lower_upper(y_bin, fbetween_method,
                                          fbetween_args, statistic)

        if fbetween_kwargs is None:
            fbetween_kwargs = {}

        if "color" not in fbetween_kwargs:
            fbetween_kwargs["color"] = color
        if "alpha" not in fbetween_kwargs:
            fbetween_kwargs["alpha"] = 0.2

        _fbt_low, _fbt_up = fbetween

        if fbetween_bad_policy == 'omit':
            _x_bin = x_bin
        elif fbetween_bad_policy == 'skip':
            _x_bin, _fbt_low, _fbt_up = remove_bad([x_bin, _fbt_low, _fbt_up])
        elif fbetween_bad_policy == 'zero':
            is_bad = flag_bad(_fbt_low) | flag_bad(_fbt_up)
            _fbt_low[is_bad] = 0
            _fbt_up[is_bad] = 0
        else:
            raise ValueError(f"{fbetween_bad_policy} is not supported")

        ax.fill_between(_x_bin, _fbt_low, _fbt_up, **fbetween_kwargs)


def _trend_get_y_statistic(method, args):
    y_statistic = []

    if method is None:
        return y_statistic, args

    if method in TREND_QUANTILE_ALIASES:
        if args is None:
            args = (0.16, 0.84)
        q_low, q_up = args
        y_statistic += [f'q:{q_low}', f'q:{q_up}']
    elif method in TREND_STD_ALIASES:
        if args is None:
            args = 1
        y_statistic += ['std']
    elif method in TREND_IQR_ALIASES:
        if args is None:
            args = (0.25, 0.75, 1.5)
        q_low, q_up, _ = args
        y_statistic += [f'q:{q_low}', f'q:{q_up}']
    elif method in TREND_STD_MEAN_ALIASES:
        if args is None:
            args = 1
        y_statistic += ['std_mean']
    elif method in TREMD_STD_MEDIAN_ALIASES:
        if args is None:
            args = 1
        y_statistic += ['std_median']
    else:
        raise ValueError(f"{method} is not supported")

    return y_statistic, args


def _trend_get_lower_upper(y_bin, method, args, statistic):

    # y_bin is needed for TREND_STD_ALIASES, TREND_STD_MEAN_ALIASES, and TREMD_STD_MEDIAN_ALIASES

    if method in TREND_QUANTILE_ALIASES:
        q_low, q_up = args
        return statistic[f'y_q:{q_low}'], statistic[f'y_q:{q_up}']
    elif method in TREND_STD_ALIASES:
        _d = args * statistic['y_std']
        return y_bin - _d, y_bin + _d
    elif method in TREND_IQR_ALIASES:
        q_low, q_up, m = args
        _low, _up = statistic[f'y_q:{q_low}'], statistic[f'y_q:{q_up}']
        iqr = _up - _low
        return q_low - m * iqr, q_up + m * iqr
    elif method in TREND_STD_MEAN_ALIASES:
        _d = args * statistic['y_std_mean']
        return y_bin - _d, y_bin + _d
    elif method in TREMD_STD_MEDIAN_ALIASES:
        _d = args * statistic['y_std_median']
        return y_bin - _d, y_bin + _d
    else:
        raise ValueError(f"{method} is not supported")


def _trend_interval(ax, x_edges, x_bin, y_bin, statistic, color, plot_kwargs,
                    yerr_method, yerr_args, errorbar_kwargs, fbetween_method,
                    fbetween_args, fbetween_kwargs):

    if y_bin is not None:

        if plot_kwargs is None:
            plot_kwargs = {}

        if "color" not in plot_kwargs:
            plot_kwargs["color"] = color

        for i, this_y in enumerate(y_bin):
            if not (np.isnan(this_y) or np.isinf(this_y)):
                x_min, x_max = x_edges[i:i + 2]
                ax.plot([x_min, x_max], [this_y, this_y], **plot_kwargs)

    if yerr_method is not None:

        if y_bin is not None:

            yerr_low, yerr_up = _trend_get_lower_upper(y_bin, yerr_method,
                                                       yerr_args, statistic)

            if x_bin is None:
                x_bin = (x_edges[:-1] + x_edges[1:]) / 2
            xerr = (x_bin - x_edges[:-1], x_edges[1:] - x_bin)
            yerr = (y_bin - yerr_low, yerr_up - y_bin)

            if errorbar_kwargs is None:
                errorbar_kwargs = {}

            if "color" not in errorbar_kwargs:
                errorbar_kwargs["color"] = color
            if "linestyle" not in errorbar_kwargs:
                errorbar_kwargs["linestyle"] = ""

            for i, (this_x, this_y, this_xerr_low, this_xerr_up, this_yerr_low,
                    this_yerr_up) in enumerate(zip(x_bin, y_bin, *xerr,
                                                   *yerr)):
                if ~np.any(
                        flag_bad([
                            this_x, this_y, this_xerr_low, this_xerr_up,
                            this_yerr_low, this_yerr_up
                        ])):
                    # print(this_xerr_low, this_xerr_up)
                    ax.errorbar(this_x,
                                this_y,
                                xerr=[[this_xerr_low], [this_xerr_up]],
                                yerr=[[this_yerr_low], [this_yerr_up]],
                                **errorbar_kwargs)
        else:
            print(
                "Warning: if is_x_interval is True, y_method is required if yerr_method is set, but y_method is None, skip the yerr plot."
            )

    if fbetween_method is not None:

        fbetween = _trend_get_lower_upper(y_bin, fbetween_method,
                                          fbetween_args, statistic)

        if fbetween_kwargs is None:
            fbetween_kwargs = {}

        if "color" not in fbetween_kwargs:
            fbetween_kwargs["color"] = color
        if "alpha" not in fbetween_kwargs:
            fbetween_kwargs["alpha"] = 0.2

        _fbt_low, _fbt_up = fbetween

        for i, (this_low, this_up) in enumerate(zip(_fbt_low, _fbt_up)):
            if not np.any(flag_bad([this_low, this_up])):
                x_min, x_max = x_edges[i:i + 2]
                ax.fill_between([x_min, x_max], this_low, this_up,
                                **fbetween_kwargs)


scatter_color_cycle = ColorCycler()


def plot_scatter(x,
                 y,
                 xerr=None,
                 yerr=None,
                 z=None,
                 color=None,
                 with_colorbar=True,
                 ax=None,
                 range=None,
                 auto_p=None,
                 weights=None,
                 x_jitter=None,
                 y_jitter=None,
                 label=None,
                 is_z_kde=False,
                 kde_bw_method=None,
                 if_smooth_z=False,
                 n_smooth=0.5,
                 linestyle=None,
                 line_kwargs=None,
                 errorbar_kwargs=None,
                 **kwargs):
    '''
    KDE for z is inspired by the plots of Yunning Zhao (赵韵宁).
    '''

    # TODO: line ordered
    # TODO: z_range, automatically adjust?

    x, y, z, weights, xerr, yerr = all_asarray([x, y, z, weights, xerr, yerr])

    has_z = False

    if z is None:
        z = np.ones_like(x)
        if is_z_kde:
            has_z = True
    else:
        has_z = True
        if is_z_kde:
            is_z_kde = False
            print("Warning: is_z_kde is ignored when z is provided")

    if has_z and color is not None:
        print(
            "Warning: color is ignored when z is provided and is_z_kde is True"
        )

    if color is None:
        # load color from plt.rcParams
        color = scatter_color_cycle.next(ax=ax)

    if weights is None:
        weights = np.ones_like(x)

    if (xerr is not None) or (yerr is not None):
        has_err = True
        if errorbar_kwargs is None:
            errorbar_kwargs = {}
        if "fmt" not in errorbar_kwargs:
            errorbar_kwargs["fmt"] = ""
        if "linestyle" not in errorbar_kwargs:
            errorbar_kwargs["linestyle"] = ""
        if "cmap" not in errorbar_kwargs:
            errorbar_kwargs["cmap"] = kwargs.get("cmap", None)
        if "norm" not in errorbar_kwargs:
            errorbar_kwargs["norm"] = kwargs.get("norm", None)
        if "alpha" not in errorbar_kwargs:
            errorbar_kwargs["alpha"] = kwargs.get("alpha", None)
        if "vmin" not in errorbar_kwargs:
            errorbar_kwargs["vmin"] = kwargs.get("vmin", None)
        if "vmax" not in errorbar_kwargs:
            errorbar_kwargs["vmax"] = kwargs.get("vmax", None)
    else:
        has_err = False

    if linestyle is not None:
        has_line = True
        if line_kwargs is None:
            line_kwargs = {}
        if "fmt" not in errorbar_kwargs:
            errorbar_kwargs["fmt"] = ""
        if "alpha" not in errorbar_kwargs:
            errorbar_kwargs["alpha"] = kwargs.get("alpha", None)
    else:
        has_line = False

    if ax is None:
        ax = plt.gca()

    x, y, z, weights, xerr, yerr = remove_bad([x, y, z, weights, xerr, yerr],
                                              to_transpose=[4, 5])

    if any_empty([x, y, z, weights, xerr, yerr]):
        warnings.warn(
            "x, y, z, weights, xerr, or yerr are empty after remove bad data, skip the plot"
        )
        return

    range = auto_set_range(x, y, range, auto_p)

    x_range = range[0]
    y_range = range[1]

    is_in_range = (x >= x_range[0]) & (x <= x_range[1]) & (y >= y_range[0]) & (
        y <= y_range[1])

    _x = x[is_in_range]
    _y = y[is_in_range]
    _weights = weights[is_in_range]

    if has_err:
        # can not use _xerr = xerr[:, is_in_range]
        # because xerr can be a 1-D array
        _xerr = xerr.T[is_in_range].T if xerr is not None else None
        _yerr = yerr.T[is_in_range].T if yerr is not None else None

    if is_z_kde:
        X = np.vstack([_x, _y])
        _z = gaussian_kde(X, bw_method=kde_bw_method, weights=_weights)(X)
    else:
        _z = z[is_in_range]

    if has_z:
        if if_smooth_z:
            _z = loess_2d_map(_x, _y, _z, _x, _y, _weights, n_smooth)

        _x = jitter_data(_x, x_jitter)
        _y = jitter_data(_y, y_jitter)

        sc = ax.scatter(_x, _y, c=_z, label=label, **kwargs)
        if with_colorbar:
            plt.colorbar(sc, ax=ax)
        if has_err:
            plot_errorbar(_x,
                          _y,
                          c=_z,
                          xerr=_xerr,
                          yerr=_yerr,
                          ax=ax,
                          with_colorbar=False,
                          **errorbar_kwargs)
        if has_line:
            # TODO: somehow derive the color from _z and cmap?
            ax.plot(_x, _y, linestyle=linestyle, color=color, **line_kwargs)
    else:

        _x = jitter_data(_x, x_jitter)
        _y = jitter_data(_y, y_jitter)

        # c = color makes facecolors='none' does not work
        ax.scatter(_x, _y, color=color, label=label, **kwargs)
        if has_err:
            plot_errorbar(_x,
                          _y,
                          color=color,
                          xerr=_xerr,
                          yerr=_yerr,
                          ax=ax,
                          with_colorbar=False,
                          **errorbar_kwargs)
        if has_line:
            ax.plot(_x, _y, linestyle=linestyle, color=color, **line_kwargs)


def plot_corner(xs,
                bins=20,
                range=None,
                weights=None,
                color="k",
                hist_bin_factor=1,
                kde_smooth=False,
                kde_smooth1d=False,
                smooth=None,
                smooth1d=None,
                labels=None,
                label_kwargs=None,
                show_titles=False,
                title_fmt=".2f",
                title_kwargs=None,
                truths=None,
                truth_color="#4682b4",
                scale_hist=False,
                quantiles=None,
                verbose=False,
                fig=None,
                max_n_ticks=5,
                top_ticks=False,
                use_math_text=False,
                reverse=False,
                hist_kwargs=None,
                plot_add=None,
                plot_add_1d=None,
                dpi=None,
                **hist2d_kwargs):
    # docstring should be copied from Bcorner.corner
    """
    The wrapper of ``Bcorner.corner``.

    Make a *sick* corner plot showing the projections of a data set in a
    multi-dimensional space. kwargs are passed to hist2d() or used for
    `matplotlib` styling.

    Parameters
    ----------
    xs : array_like[nsamples, ndim]
        The samples. This should be a 1- or 2-dimensional array. For a 1-D
        array this results in a simple histogram. For a 2-D array, the zeroth
        axis is the list of samples and the next axis are the dimensions of
        the space.

    bins : int or array_like[ndim,]
        The number of bins to use in histograms, either as a fixed value for
        all dimensions, or as a list of integers for each dimension.

    weights : array_like[nsamples,]
        The weight of each sample. If `None` (default), samples are given
        equal weight.

    color : str
        A ``matplotlib`` style color for all histograms.

    hist_bin_factor : float or array_like[ndim,]
        This is a factor (or list of factors, one for each dimension) that
        will multiply the bin specifications when making the 1-D histograms.
        This is generally used to increase the number of bins in the 1-D plots
        to provide more resolution.

    smooth, smooth1d : float
       The standard deviation for Gaussian kernel passed to
       `scipy.ndimage.gaussian_filter` to smooth the 2-D and 1-D histograms
       respectively. If `None` (default), no smoothing is applied.

    labels : iterable (ndim,)
        A list of names for the dimensions. If a ``xs`` is a
        ``pandas.DataFrame``, labels will default to column names.

    label_kwargs : dict
        Any extra keyword arguments to send to the `set_xlabel` and
        `set_ylabel` methods.

    show_titles : bool
        Displays a title above each 1-D histogram showing the 0.5 quantile
        with the upper and lower errors supplied by the quantiles argument.

    title_fmt : string
        The format string for the quantiles given in titles. If you explicitly
        set ``show_titles=True`` and ``title_fmt=None``, the labels will be
        shown as the titles. (default: ``.2f``)

    title_kwargs : dict
        Any extra keyword arguments to send to the `set_title` command.

    range : iterable (ndim,)
        A list where each element is either a length 2 tuple containing
        lower and upper bounds or a float in range (0., 1.)
        giving the fraction of samples to include in bounds, e.g.,
        [(0.,10.), (1.,5), 0.999, etc.].
        If a fraction, the bounds are chosen to be equal-tailed.

    truths : iterable (ndim,)
        A list of reference values to indicate on the plots.  Individual
        values can be omitted by using ``None``.

    truth_color : str
        A ``matplotlib`` style color for the ``truths`` makers.

    scale_hist : bool
        Should the 1-D histograms be scaled in such a way that the zero line
        is visible?

    quantiles : iterable
        A list of fractional quantiles to show on the 1-D histograms as
        vertical dashed lines.

    verbose : bool
        If true, print the values of the computed quantiles.

    plot_contours : bool
        Draw contours for dense regions of the plot.

    use_math_text : bool
        If true, then axis tick labels for very large or small exponents will
        be displayed as powers of 10 rather than using `e`.

    reverse : bool
        If true, plot the corner plot starting in the upper-right corner
        instead of the usual bottom-left corner

    max_n_ticks: int
        Maximum number of ticks to try to use

    top_ticks : bool
        If true, label the top ticks of each axis

    fig : matplotlib.Figure
        Overplot onto the provided figure object.

    hist_kwargs : dict
        Any extra keyword arguments to send to the 1-D histogram plots.

    **hist2d_kwargs
        Any remaining keyword arguments are sent to `corner.hist2d` to generate
        the 2-D histogram plots.

    """
    return corner(xs,
                  bins=bins,
                  range=range,
                  weights=weights,
                  color=color,
                  hist_bin_factor=hist_bin_factor,
                  kde_smooth=kde_smooth,
                  kde_smooth1d=kde_smooth1d,
                  smooth=smooth,
                  smooth1d=smooth1d,
                  labels=labels,
                  label_kwargs=label_kwargs,
                  show_titles=show_titles,
                  title_fmt=title_fmt,
                  title_kwargs=title_kwargs,
                  truths=truths,
                  truth_color=truth_color,
                  scale_hist=scale_hist,
                  quantiles=quantiles,
                  verbose=verbose,
                  fig=fig,
                  max_n_ticks=max_n_ticks,
                  top_ticks=top_ticks,
                  use_math_text=use_math_text,
                  reverse=reverse,
                  hist_kwargs=hist_kwargs,
                  plot_add=plot_add,
                  plot_add_1d=plot_add_1d,
                  dpi=dpi,
                  **hist2d_kwargs)


def plot_contour(x,
                 y,
                 bins=20,
                 range='auto',
                 kde_smooth=False,
                 auto_p=None,
                 weights=None,
                 levels=5,
                 smooth=None,
                 ax=None,
                 color=None,
                 quiet=False,
                 plot_datapoints=True,
                 plot_density=True,
                 plot_contours=True,
                 no_fill_contours=False,
                 fill_contours=False,
                 contour_kwargs=None,
                 contourf_kwargs=None,
                 data_kwargs=None,
                 pcolor_kwargs=None):
    # docstring should be copied from Bcorner.hist2d
    """
    The wrapper of ``Bcorner.hist2d``.

    Plot a 2-D histogram of samples.

    Parameters
    ----------
    x : array_like[nsamples,]
       The samples.

    y : array_like[nsamples,]
       The samples.

    range: array_like[2, 2] or string
       ([x_min, x_max], [y_min, y_max]), if not 'auto'
       The range is automatically determined according to quantile specified by auto_p, if 'auto'
       default: 'auto'

    kde_smooth: if use kde smooth

    auto_p: array_like[2, 2] or string
       Used to generate range if range == 'auto'
       x_min = np.percentile(x, auto_p[0][0])
       x_max = np.percentile(x, auto_p[0][1])
       y_min = np.percentile(y, auto_p[1][0])
       y_max = np.percentile(y, auto_p[1][1])
       default: ([1, 99], [1, 99])
    
    weights : Optional[array_like[nsamples,]]
        An optional weight corresponding to each sample.

    levels : array_like
        The contour levels to draw.

    smooth : float
        The standard deviation of the Gaussian kernel used to smooth the
        density values. If ``None``, no smoothing is applied.

    ax : matplotlib.Axes
        A axes instance on which to add the 2-D histogram.

    color : str
        The color of the datapoints, density map, and contours.

    quiet : bool
        If true, suppress warnings for small datasets.

    plot_datapoints : bool
        Draw the individual data points.

    plot_density : bool
        Draw the density colormap.

    plot_contours : bool
        Draw the contours.

    no_fill_contours : bool
        Add no filling at all to the contours (unlike setting
        ``fill_contours=False``, which still adds a white fill at the densest
        points).

    fill_contours : bool
        Fill the contours.

    contour_kwargs : dict
        Any additional keyword arguments to pass to the `contour` method.

    contourf_kwargs : dict
        Any additional keyword arguments to pass to the `contourf` method.

    data_kwargs : dict
        Any additional keyword arguments to pass to the `plot` method when
        adding the individual data points.

    pcolor_kwargs : dict
        Any additional keyword arguments to pass to the `pcolor` method when
        adding the density colormap.
    """
    hist2d(x,
           y,
           bins=bins,
           range=range,
           kde_smooth=kde_smooth,
           auto_p=auto_p,
           weights=weights,
           levels=levels,
           smooth=smooth,
           ax=ax,
           color=color,
           quiet=quiet,
           plot_datapoints=plot_datapoints,
           plot_density=plot_density,
           plot_contours=plot_contours,
           no_fill_contours=no_fill_contours,
           fill_contours=fill_contours,
           contour_kwargs=contour_kwargs,
           contourf_kwargs=contourf_kwargs,
           data_kwargs=data_kwargs,
           pcolor_kwargs=pcolor_kwargs)


def plot_heatmap(x,
                 y,
                 z,
                 weights=None,
                 ax=None,
                 bins=10,
                 min_data=0,
                 range=None,
                 auto_p=None,
                 map_kind='pcolor',
                 pcolor_kwargs=None,
                 contour_kwargs=None,
                 set_clabel=False,
                 clabel_kwargs=None):  # sourcery skip: avoid-builtin-shadow
    """
    kind: 'pcolor' or 'contour'
    """
    # TODO: z_range
    # TODO: weights

    bad = flag_bad(x) | flag_bad(y) | flag_bad(z)
    x = x[~bad]
    y = y[~bad]
    z = z[~bad]

    if is_empty(x) or is_empty(y) or is_empty(z):
        warnings.warn(
            "The x, y or z are empty after remove bad data skip the plot")
        return

    # TODO: z range
    range = auto_set_range(x, y, range, auto_p)

    X, Y, Z, x_edges, y_edges = bin_2d(x,
                                       y,
                                       z,
                                       bins,
                                       min_data=min_data,
                                       range=range)
    if ax is None:
        ax = plt.gca()

    if map_kind == 'pcolor':
        if pcolor_kwargs is None:
            pcolor_kwargs = {}

        # Maybe use pcolormesh for high performance?
        # https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.pcolormesh.html#differences-pcolor-pcolormesh
        ax.pcolor(x_edges, y_edges, Z, **pcolor_kwargs)

    elif map_kind == 'contour':

        if contour_kwargs is None:
            contour_kwargs = {}
        cont = ax.contour(X, Y, Z, **contour_kwargs)
        if set_clabel:
            if clabel_kwargs is None:
                clabel_kwargs = {}
            ax.clabel(cont, **clabel_kwargs)

        return cont

    elif map_kind == 'contourf':
        if contour_kwargs is None:
            contour_kwargs = {}
        cont = ax.contourf(X, Y, Z, **contour_kwargs)
        if set_clabel:
            if clabel_kwargs is None:
                clabel_kwargs = {}
            ax.clabel(cont, **clabel_kwargs)

        return cont


def plot_sample_to_point(x,
                         y,
                         ax=None,
                         weights=None,
                         ddof=0,
                         center_type='mean',
                         error_type='std',
                         quantiles=None,
                         errorbar_kwargs=None):
    """
    The function to summarize a sample to a point and then plot it

    Parameters:
    - x: array-like
        The x-coordinates of the sample.
    - y: array-like
        The y-coordinates of the sample.
    - ax: matplotlib.axes.Axes, optional
        The axes on which to plot the summarized point. If not provided, the current axes will be used.
    - weights: array-like, optional
        An array of weights associated with each data point. If not provided, all data points are assumed to have equal weight.
    - ddof: int, optional
        The delta degrees of freedom used in the calculation of standard deviation. Default is 0.
    - center_type: str, optional
        The method used to calculate the center point. Can be either 'mean' or 'median'. Default is 'mean'.
    - error_type: str, optional
        The method used to calculate the error bars. Can be 'std', 'std_mean', or 'quantile'. Default is 'std'.
    - quantiles: array-like, optional
        The quantiles used to calculate the error bars when error_type is 'quantile'. Default is None.
    - errorbar_kwargs: dict, optional
        Additional keyword arguments to be passed to the errorbar function.

    Raises:
    - ValueError: If center_type or error_type is not one of the specified options.

    Returns:
    - None

    """
    bad = flag_bad(x) | flag_bad(y)
    x = x[~bad]
    y = y[~bad]

    if ax is None:
        ax = plt.gca()

    if errorbar_kwargs is None:
        errorbar_kwargs = {}

    if center_type == 'mean':
        weighted_stats = DescrStatsW(np.c_[x, y], weights=weights, ddof=ddof)
        x_cen = weighted_stats.mean[0]
        y_cen = weighted_stats.mean[1]
    elif center_type == 'median':
        x_cen = quantile(x, q=0.5, weights=weights)[0]
        y_cen = quantile(y, q=0.5, weights=weights)[0]
    else:
        raise ValueError('center_type must be one of mean or median')

    if error_type == 'std':
        weighted_stats = DescrStatsW(np.c_[x, y], weights=weights, ddof=ddof)
        x_err = weighted_stats.std[0]
        y_err = weighted_stats.std[1]
    elif error_type == 'std_mean':
        weighted_stats = DescrStatsW(np.c_[x, y], weights=weights, ddof=ddof)
        x_err = weighted_stats.std_mean[0]
        y_err = weighted_stats.std_mean[1]
    elif error_type == 'quantile':
        if quantiles is None:
            quantiles = [0.16, 0.84]
        x_quant = quantile(x, q=quantiles, weights=weights).reshape(-1, 1)
        y_quant = quantile(y, q=quantiles, weights=weights).reshape(-1, 1)
        x_err = np.abs(x_quant - x_cen)
        y_err = np.abs(y_quant - y_cen)
    else:
        raise ValueError('error_type must be one of std, std_mean or quantile')

    ax.errorbar(x_cen, y_cen, xerr=x_err, yerr=y_err, **errorbar_kwargs)


def plot_log_line(p1=None,
                  p2=None,
                  k=None,
                  b=None,
                  ax=None,
                  range=None,
                  N=1000,
                  is_input_log=True,
                  **kwargs):

    if ax is None:
        ax = plt.gca()

    if not is_input_log:
        p1 = (np.log10(p1[0]), np.log10(p1[1])) if p1 is not None else None
        p2 = (np.log10(p2[0]), np.log10(p2[1])) if p2 is not None else None

    if range is None:
        range = ax.get_xlim()

    xx = np.linspace(range[0], range[1], N)

    if (k is None) or (b is None):
        if p1 is not None:
            if p2 is not None:
                k = (p2[1] - p1[1]) / (p2[0] - p1[0])
                b = p1[1] - k * p1[0]
            elif k is not None:
                b = p1[1] - k * p1[0]
            elif b is not None:
                k = (p1[1] - b) / p1[0]
            else:
                raise ValueError("Only p1 is provided, need p2 or k or b")
        else:
            raise ValueError("No input provided")

    yy = 10**(k * np.log10(xx) + b)
    ax.plot(xx, yy, **kwargs)


def plot_line(x=None,
              y=None,
              p1=None,
              p2=None,
              k=None,
              b=None,
              ax=None,
              **kwargs):
    """
    Plot a line on the given axes.

    Parameters:
    - x: float, optional - The x-coordinate where the vertical line should be plotted.
    - y: float, optional - The y-coordinate where the horizontal line should be plotted.
    - p1: tuple, optional - The coordinates of the first point on the line.
    - p2: tuple, optional - The coordinates of the second point on the line.
    - k: float, optional - The slope of the line.
    - b: float, optional - The y-intercept of the line.
    - ax: matplotlib.axes.Axes, optional - The axes on which to plot the line.
    - **kwargs: Additional keyword arguments to be passed to `ax.axline`.

    Raises:
    - ValueError: If the input is invalid.

    Returns:
    - None
    """
    if ax is None:
        ax = plt.gca()

    if x is not None:
        ax.axvline(x, **kwargs)
        return

    if y is not None:
        ax.axhline(y, **kwargs)
        return

    if p1 is not None:
        if p2 is not None:
            ax.axline(p1, p2, **kwargs)
            return
        elif k is not None:
            ax.axline(p1, slope=k, **kwargs)
            return
        elif b is not None:
            ax.axline(p1, slope=(p1[1] - b) / p1[0], **kwargs)
            return

    # sourcery skip: merge-nested-ifs
    if k is not None:
        if b is not None:
            ax.axline((0, b), slope=k, **kwargs)
            return

    raise ValueError("Invalid input")


def imshow(X, ax=None, mask=None, **kwargs):

    if ax is None:
        ax = plt.gca()

    if mask is not None:
        X = X.astype(float).copy()
        X[~mask] = np.nan

    return ax.imshow(X, **kwargs)


# TODO: consider P(p|k, N)
# TODO: use plt.hist, only add errorbar if needed
# TODO: use plt.bar only for color coding and histtype == 'bar'
def plot_hist(x,
              bins=10,
              range=None,
              weights=None,
              ax=None,
              density=False,
              interval="frequentist-confidence",
              sigma=1,
              background=0,
              confidence_level=None,
              return_data=False,
              hide_zero_errorbar=False,
              **kwargs):
    """
    Plot the histogram of x
    """
    if ax is None:
        ax = plt.gca()

    x = remove_bad([x])[0]

    centers, N, lower, upper, edges, d_bin = get_epdf(
        x,
        bins=bins,
        range=range,
        weights=weights,
        density=density,
        interval=interval,
        sigma=sigma,
        background=background,
        confidence_level=confidence_level)

    yerr_low = N - lower
    yerr_up = upper - N

    if hide_zero_errorbar:
        yerr_low[yerr_low == 0] = np.nan
        yerr_up[yerr_up == 0] = np.nan

    ax.bar(centers, N, width=d_bin, yerr=[yerr_low, yerr_up], **kwargs)

    if return_data:
        return {
            "N": N,
            "edges": edges,
            "centers": centers,
            "lower": lower,
            "upper": upper
        }


# TODO: single err for all
# TODO: different upper and lower err
def plot_errorbar(x,
                  y,
                  xerr=None,
                  yerr=None,
                  c=None,
                  ax=None,
                  vmin=None,
                  vmax=None,
                  cmap=None,
                  norm=None,
                  with_colorbar=False,
                  **kwargs):
    """
    Plot error bars with color coding based on a third variable 'c'.

    Parameters:
        x : array-like
            The x-coordinates.
        y : array-like
            The y-coordinates.
        c : array-like, optional
            The values used to color the error bars. If None, the error bars will not be color-coded.
        cmap : str or Colormap, optional
            The colormap used to map the 'c' values to colors. Default is 'viridis'.
        with_colorbar : bool, optional
            Whether to display a colorbar next to the plot. Default is False.
        **kwargs : additional arguments
            Additional arguments to pass to plt.errorbar.

    Returns:
        matplotlib.cm.ScalarMappable
            A ScalarMappable object that can be used to create a colorbar.
    """

    x = np.asarray(x)
    y = np.asarray(y)

    if ax is None:
        ax = plt.gca()

    if c is None:
        return ax.errorbar(x, y, xerr=xerr, yerr=yerr, **kwargs)
    else:
        c = np.asarray(c)

    # Create a colormap object
    if cmap is None:
        # get default colormap from rcParams
        cmap = plt.rcParams['image.cmap']

    if isinstance(cmap, str):
        cmap = plt.colormaps.get_cmap(cmap)

    if vmin is None:
        vmin = c.min()
    if vmax is None:
        vmax = c.max()

    if norm is None:
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
    elif isinstance(norm, str):
        norm = scale_factory(norm, c, vmin=vmin, vmax=vmax)

    colors = cmap(norm(c))

    # Plotting with error bars and color coding
    if xerr is None:
        xerr = [None] * len(x)
    if yerr is None:
        yerr = [None] * len(y)

    for i in range(len(x)):  # pylint: disable=consider-using-enumerate
        ax.errorbar(x[i],
                    y[i],
                    xerr=xerr[i],
                    yerr=yerr[i],
                    color=colors[i],
                    **kwargs)

    # Create a ScalarMappable and an axes-level colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)

    if with_colorbar:
        plt.colorbar(sm, ax=ax)

    return ax, sm


def plot_volcano(x_lst, y, method='pearsonr', ax=None, **kwargs):
    if ax is None:
        ax = plt.gca()
    corr_dict = {'pvalue': [], 'statistic': []}
    for x in x_lst:
        this_res = get_correlation_coefficients(x, y)[method]
        corr_dict['pvalue'].append(this_res.pvalue)
        corr_dict['statistic'].append(this_res.statistic)
    ax.scatter(corr_dict['statistic'], -np.log10(corr_dict['pvalue']),
               **kwargs)
    ax.set_xlim(-1, 1)
    return ax


def ax_fill_between(y1, y2, ax=None, **kwargs):

    if ax is None:
        ax = plt.gca()

    this_xlim = ax.get_xlim()
    ax.fill_between(this_xlim, [y1, y1], [y2, y2], **kwargs)
    ax.set_xlim(this_xlim)


def ax_fill_betweenx(x1, x2, ax=None, **kwargs):

    if ax is None:
        ax = plt.gca()

    this_ylim = ax.get_ylim()
    ax.fill_betweenx(this_ylim, [x1, x1], [x2, x2], **kwargs)
    ax.set_ylim(this_ylim)
