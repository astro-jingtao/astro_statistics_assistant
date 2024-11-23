import warnings

import matplotlib.pyplot as plt
from matplotlib.scale import scale_factory
import numpy as np
from scipy.stats import gaussian_kde
from statsmodels.stats.weightstats import DescrStatsW

from .Bcorner import corner, hist2d, quantile
from .binning_methods import bin_1d, bin_2d, get_epdf
from .loess2d import loess_2d_map
from .utils import any_empty, auto_set_range, flag_bad, is_empty, remove_bad, all_asarray

# TODO: extract common code
# - flag and remove bad


# TODO: skind: quantile or sigma
def plot_trend(x,
               y,
               bins=20,
               ytype='median',
               ax=None,
               range=None,
               auto_p=None,
               weights=None,
               N_min=1,
               lowlim=25,
               uplim=75,
               fkind=None,
               plot_line=True,
               prop_kwargs=None,
               errorbar_kwargs=None,
               fbetween_kwargs=None,
               plot_kwargs=None):  # sourcery skip: avoid-builtin-shadow
    """
    Make a plot to show the trend between x and y

    Parameters
    -----------------
    x : array_like[nsamples,]                                     
        The samples.                             
                                                           
    y : array_like[nsamples,]                            
        The samples.
                                                                     
    ytype: Character string or float
        The y value used to plot
        The available character string is "median" or 'mean'. If ytype is set as "median", the trend is shown by the median value of y as a function of x.
        if ytype is float, y_value = np.percentile(y, ytype)
        default: "median"


    range: array_like[2, 2] or string ([x_min, x_max], [y_min, y_max]), 
        if not 'auto', the range is automatically determined according to quantile specified by auto_p, if 'auto'                                                        
        default: 'auto'

    auto_p: array_like[2, 2] or string
       Used to generate range if range == 'auto'
       x_min = np.percentile(x, auto_p[0][0])
       x_max = np.percentile(x, auto_p[0][1])
       y_min = np.percentile(y, auto_p[1][0])
       y_max = np.percentile(y, auto_p[1][1])
       default: ([1, 99], [1, 99])

    weights: Optional[array_like[nsamples,]] 
        An optional weight corresponding to each sample.

    N_min: int
        The minimum number of samples in each bin to plot the trend
    
    ax : matplotlib.Axes
        A axes instance on which to add the line.

    ifscatter: whether to plot scatter

    uplim (%): The upper limit of the scatter, in [0, 100]. 

    lowlim (%): The lower limit of the scatter, in [0, 100]

    fkind: which ways to show the scatter, "errorbar" and "fbetween" are available

    plot_line: whether to plot the line, only valid when fkind is "fbetween"
        
    plot_kwargs: function in ``matplotlib``


    prop_kwargs: dict (to be added)
        The extra property used to constrain the x, y, data
        props : array_like[nsamples,]                                     
            The samples with size same as x/y.                             
        pmax : the maximum value of props
        pmin : the minimum value of props
        
    plot_scatter_kwargs: dict
        to describe the scatter
        function in ``matplotlib``
    
    """

    if ax is None:
        ax = plt.gca()

    if plot_kwargs is None:
        plot_kwargs = {}

    if prop_kwargs is not None:
        props = prop_kwargs["props"]
        pmin = prop_kwargs.get("pmin", min(props))
        pmax = prop_kwargs.get("pmax", max(props))
        prop_index = (props >= pmin) & (props <= pmax)
        x = x[prop_index]
        y = y[prop_index]
        # print(np.shape(x), np.shape(y))

    ifscatter = fkind is not None

    bad = flag_bad(x) | flag_bad(y)
    x = x[~bad]
    y = y[~bad]

    if is_empty(x) or is_empty(y):
        warnings.warn(
            "The x or y are empty after remove bad data skip the plot")
        return

    range = auto_set_range(x, y, range, auto_p)

    xrange = range[0]
    yrange = range[1]

    if weights is None:
        weights = np.ones_like(x)

    is_y_in_range = (y >= yrange[0]) & (y <= yrange[1])

    y_statistic = [ytype]
    if ifscatter:
        low_name = f"q:{lowlim/100:.2f}"
        up_name = f"q:{uplim/100:.2f}"

        y_statistic.append(low_name)
        y_statistic.append(up_name)

    _, _, _, statistic = bin_1d(x[is_y_in_range],
                                y[is_y_in_range],
                                weights=weights[is_y_in_range],
                                x_statistic=['median'],
                                y_statistic=y_statistic,
                                bins=bins,
                                range=xrange,
                                min_data=N_min)

    # TODO: support the error of median or mean
    if ifscatter:
        if fkind == "errorbar":
            if errorbar_kwargs is None:
                errorbar_kwargs = {}
                errorbar_kwargs["color"] = plot_kwargs.get("color")
                errorbar_kwargs["label"] = plot_kwargs.get("label")
            else:
                if "color" not in errorbar_kwargs:
                    errorbar_kwargs["color"] = plot_kwargs.get("color")
                if "label" not in errorbar_kwargs:
                    errorbar_kwargs["label"] = plot_kwargs.get("label")
            # TODO: deal with ytype is mean and yerr is negative
            ax.errorbar(
                statistic['x_median'],
                statistic[f'y_{ytype}'],
                yerr=(statistic[f'y_{ytype}'] - statistic[f'y_{low_name}'],
                      statistic[f'y_{up_name}'] - statistic[f'y_{ytype}']),
                **errorbar_kwargs)
        elif fkind == "fbetween":
            if plot_line:
                ax.plot(statistic['x_median'], statistic[f'y_{ytype}'],
                        **plot_kwargs)
            if fbetween_kwargs is None:
                fbetween_kwargs = {}
                fbetween_kwargs["color"] = plot_kwargs.get("color", "r")
                fbetween_kwargs["alpha"] = 0.2
            else:
                if "color" not in fbetween_kwargs:
                    fbetween_kwargs["color"] = plot_kwargs.get("color", "r")
                if "alpha" not in fbetween_kwargs:
                    fbetween_kwargs["alpha"] = 0.2
            ax.fill_between(statistic['x_median'], statistic[f'y_{up_name}'],
                            statistic[f'y_{low_name}'], **fbetween_kwargs)
    else:
        ax.plot(statistic['x_median'], statistic[f'y_{ytype}'], **plot_kwargs)


def plot_scatter(x,
                 y,
                 xerr=None,
                 yerr=None,
                 z=None,
                 color=None,
                 ax=None,
                 range=None,
                 auto_p=None,
                 weights=None,
                 label=None,
                 is_z_kde=False,
                 kde_bw_method=None,
                 if_smooth_z=False,
                 n_smooth=0.5,
                 errorbar_kwargs=None,
                 **kwargs):

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
        print("Warning: c is ignored when z is provided and is_z_kde is True")

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

    if ax is None:
        ax = plt.gca()

    x, y, z, weights, xerr, yerr = remove_bad([x, y, z, weights, xerr, yerr])

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
    if is_z_kde:
        X = np.vstack([_x, _y])
        _z = gaussian_kde(X, bw_method=kde_bw_method, weights=_weights)(X)
    else:
        _z = z[is_in_range]

    if has_z:
        if if_smooth_z:
            _z = loess_2d_map(_x, _y, _z, _x, _y, _weights, n_smooth)

        sc = ax.scatter(_x, _y, c=_z, label=label, **kwargs)
        plt.colorbar(sc, ax=ax)
        if has_err:
            plot_errorbar(_x,
                          _y,
                          c=_z,
                          xerr=xerr,
                          yerr=yerr,
                          ax=ax,
                          with_colorbar=False,
                          **errorbar_kwargs)
    else:
        ax.scatter(_x, _y, c=color, label=label, **kwargs)
        if has_err:
            print(errorbar_kwargs)
            plot_errorbar(_x,
                          _y,
                          color=color,
                          xerr=xerr,
                          yerr=yerr,
                          ax=ax,
                          with_colorbar=False,
                          **errorbar_kwargs)


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
    corner(xs,
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

    ax.imshow(X, **kwargs)


# TODO: consider P(p|k, N)
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
              **kwargs):
    """
    Plot the histogram of x
    """
    if ax is None:
        ax = plt.gca()

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

    ax.bar(centers, N, width=d_bin, yerr=[N - lower, upper - N], **kwargs)

    if return_data:
        return {
            "N": N,
            "edges": edges,
            "centers": centers,
            "lower": lower,
            "upper": upper
        }


# TODO: support give ax
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
        return plt.errorbar(x, y, xerr=xerr, yerr=yerr, **kwargs)
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
