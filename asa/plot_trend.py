import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic

from .utils import flag_bad


def plot_trend(x,
               y,
               bins=20,
               ytype='median',
               fig=None,
               ax=None,
               ranges=None,
               auto_p=None,
               prop_kwargs=None,
               scatter_kwargs=None,
               plot_kwargs=None):
    # TODO: support weights
    # TODO: better default values
    # TODO: bottom -> low

    """
    Make a plot to show the trend between x and y

    Parameters
    -----------------
    x : array_like[nsamples,]                                     
        The samples.                             
                                                           
    y : array_like[nsamples,]                            
        The samples.
                                                                     
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

    ytype: Character string or float
        The y value used to plot
        The available character string is "median" or 'mean'. If ytype is set as "median", the trend is shown by the median value of y as a function of x.
        if ytype is float, y_value = np.percentile(y, ytype)
        default: "median"
    
    ax : matplotlib.Axes
        A axes instance on which to add the line.
        
        
    plot_kwargs: function in ``matplotlib``


    prop_kwargs: dict (to be added)
        The extra property used to constrain the x, y, data
        props : array_like[nsamples,]                                     
            The samples with size same as x/y.                             
        pmax : the maximum value of props
        pmin : the minimum value of props

    scatter_kwargs: dict (to be added)
        ifscatter: whether to plot scatter
        uplim (%): The upper limit of the scatter
        bottomlim (%): The bottom limit of the scatter
        fkind: which ways to show the scatter, "errorbar" and "fbetween" are available
        plot_scatter_kwargs: function in ``matplotlib``
        
    """
    if ax is None:
        ax = plt.gca()

    if plot_kwargs is None:
        plot_kwargs = {}

    # TODO: default value
    if scatter_kwargs is None:
        ifscatter = False
    else:
        ifscatter = scatter_kwargs["ifscatter"]
        uplim = scatter_kwargs["uplim"]
        bottomlim = scatter_kwargs["bottomlim"]
        fkind = scatter_kwargs["fkind"]
        plot_scatter_kwargs = scatter_kwargs["plot_scatter_kwargs"]

    if prop_kwargs is not None:
        props = prop_kwargs["props"]
        pmin = prop_kwargs["pmin"]
        pmax = prop_kwargs["pmax"]
        prop_index = (props >= pmin) & (props <= pmax)
        x = x[prop_index]
        y = y[prop_index]
        print(np.shape(x), np.shape(y))

    bad = flag_bad(x) | flag_bad(y)
    x = x[~bad]
    y = y[~bad]

    if ranges is None:
        xrange = [x.min(), x.max()]
        yrange = [y.min(), y.max()]
    elif ranges == 'auto':
        if auto_p is None:
            auto_p = ([1, 99], [1, 99])
        xrange = [
            np.percentile(x, auto_p[0][0]),
            np.percentile(x, auto_p[0][1])
        ]
        yrange = [
            np.percentile(y, auto_p[1][0]),
            np.percentile(y, auto_p[1][1])
        ]
    else:
        xrange = ranges[:][0]
        yrange = ranges[:][1]

    is_y_in_range = (y > yrange[0]) & (y < yrange[1])
    if ytype == "median":
        y_statistic = "median"
    elif ytype == "mean":
        y_statistic = "mean"
    else:
        y_statistic = lambda x: np.percentile(x, float(ytype))

    loads = [
        binned_statistic(x[is_y_in_range],
                         x[is_y_in_range],
                         statistic="median",
                         bins=bins,
                         range=xrange)[0]
    ]

    statistic_list = [y_statistic]

    if ifscatter:
        upper_statistic = lambda x: np.percentile(x, uplim)
        bottom_statistic = lambda x: np.percentile(x, bottomlim)

        statistic_list.append(bottom_statistic)
        statistic_list.append(upper_statistic)

    for statistic in statistic_list:
        _value, _, _ = binned_statistic(x[is_y_in_range],
                                        y[is_y_in_range],
                                        statistic=statistic,
                                        bins=bins,
                                        range=xrange)
        loads.append(_value)

    loads = np.vstack(loads).T

    ax.plot(loads[:, 0], loads[:, 1], **plot_kwargs)


    if ifscatter:
        if fkind == "errorbar":
                ax.errorbar(loads[:, 0],
                            loads[:, 1],
                            yerr=(loads[:, 2] - loads[:, 3]) / 2.0,
                            **plot_scatter_kwargs)
        elif fkind == "fbetween":
                ax.fill_between(loads[:, 0], loads[:, 3], loads[:, 2],
                                **plot_scatter_kwargs)
