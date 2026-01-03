from typing import cast

import numpy as np
from matplotlib.axes import Axes

from asa.binning_methods import get_epdf
from asa.plot_methods.plot_utils import auto_setup_ax, prepare_data
from asa.utils import remove_bad


# TODO: consider P(p|k, N)
# TODO: use plt.hist, only add errorbar if needed
# TODO: use plt.bar only for color coding and histtype == 'bar'
@auto_setup_ax
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
    ax = cast(Axes, ax)

    x = prepare_data(x, arg_names=['x'])

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
