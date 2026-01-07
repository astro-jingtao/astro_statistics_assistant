import numpy as np
from matplotlib import cm as _cm
from matplotlib import colors as _mcolors
import matplotlib.pyplot as plt

from asa.binning_methods import get_epdf, bin_1d
from asa.plot_methods.plot_utils import prepare_data


def plot_hist(x,
              *,
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
              z=None,
              z_statistic='mean',
              cmap=None,
              vmin=None,
              vmax=None,
              color=None,
              facecolor=None,
              edgecolor=None,
              ecolor=None,
              **kwargs):
    """
    Plot the histogram of x with optional color-coding by z variable.
    
    Parameters
    ----------
    x : array_like
        Input data for histogram.
    bins : int or array_like, optional
        Number of bins or bin edges. Default is 10.
    range : tuple, optional
        Lower and upper range of bins.
    weights : array_like, optional
        Weights for each data point.
    ax : Axes, optional
        Matplotlib axes object.
    density : bool, optional
        If True, normalize histogram. Default is False.
    interval : str, optional
        Confidence interval type. Default is "frequentist-confidence".
    sigma : float, optional
        Sigma level for error bars. Default is 1.
    background : float, optional
        Background for Poisson confidence interval. Default is 0.
    confidence_level : float, optional
        Confidence level for interval.
    return_data : bool, optional
        If True, return data dict. Default is False.
    hide_zero_errorbar : bool, optional
        Hide error bars that are zero. Default is False.
    z : array_like, optional
        Variable for color-coding bars via cmap.
    z_statistic : str, optional
        Statistic to compute for z in each bin. Default is 'mean'.
    cmap : str or Colormap, optional
        Colormap for z-based coloring.
    vmin, vmax : float, optional
        Min/max values for colormap normalization.
    color : color or None, optional
        Bar color. If None and z/cmap provided, uses cmap-mapped colors.
    facecolor : color or None, optional
        Bar face color. If None and z/cmap provided, uses cmap-mapped colors.
    edgecolor : color or None, optional
        Bar edge color. If None and z/cmap provided, uses cmap-mapped colors.
    ecolor : color or None, optional
        Error bar color. If None and z/cmap provided, uses cmap-mapped colors.
    **kwargs : dict
        Additional arguments passed to ax.bar().
    
    Returns
    -------
    ax : Axes
        The axes object.
    sm : ScalarMappable or None
        ScalarMappable for colorbar (when z and cmap are used).
    data : dict, optional
        If return_data=True, returns dict with N, edges, centers, lower, upper.
    """
    if ax is None:
        ax = plt.gca()

    x = prepare_data(x, arg_names=['x'])

    # bin_1d -> binned_statistic_robust -> np.histogram_bin_edges
    # get_epdf -> np.histogram -> np.histogram_bin_edges
    _bins = np.histogram_bin_edges(x, bins=bins, range=range)

    centers, N, lower, upper, edges, d_bin = get_epdf(
        x,
        bins=_bins,
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

    # Handle color mapping from z values
    sm = None
    mapped_colors = None

    if z is not None:
        # Compute per-bin statistic of z
        _, _, _, statistic = bin_1d(x,
                                    z,
                                    weights=weights,
                                    x_statistic=None,
                                    y_statistic=z_statistic,
                                    bins=_bins,
                                    quantile=False,
                                    range=range,
                                    min_data=0)
        z_val = statistic[f'y_{z_statistic}']

        # Determine vmin/vmax if not provided
        _vmin = vmin if vmin is not None else np.nanmin(z_val)
        _vmax = vmax if vmax is not None else np.nanmax(z_val)

        # Create normalization and colormap
        norm = _mcolors.Normalize(vmin=_vmin, vmax=_vmax)
        _cmap = _cm.get_cmap(cmap) if (isinstance(cmap, str)
                                       or cmap is None) else cmap

        # Map z values to colors
        mapped_colors = _cmap(norm(z_val))

        # Create ScalarMappable for colorbar
        sm = _cm.ScalarMappable(cmap=_cmap, norm=norm)

    # Determine final colors: user-provided > cmap-mapped
    final_color = color if (color is not None) else mapped_colors
    final_facecolor = facecolor if (facecolor is not None) else final_color
    final_edgecolor = edgecolor
    final_ecolor = ecolor

    # only color supports array-like
    # do not facecolor=final_facecolor
    ax.bar(centers,
           N,
           width=d_bin,
           yerr=[yerr_low, yerr_up],
           color=final_facecolor, 
           edgecolor=final_edgecolor,
           ecolor=final_ecolor,
           **kwargs)

    return_lst = [ax, sm]

    if return_data:
        return_lst.append({
            "N": N,
            "edges": edges,
            "centers": centers,
            "lower": lower,
            "upper": upper
        })

    return tuple(return_lst)
