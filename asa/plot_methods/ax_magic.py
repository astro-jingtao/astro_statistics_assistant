from typing import cast

import numpy as np
from matplotlib.axes import Axes

from asa.plot_methods.plot_utils import auto_setup_ax


@auto_setup_ax
def ax_fill_between(y1, y2, *, ax=None, **kwargs):

    ax = cast(Axes, ax)

    this_xlim = ax.get_xlim()
    ax.fill_between(this_xlim, [y1, y1], [y2, y2], **kwargs)
    ax.set_xlim(this_xlim)


@auto_setup_ax
def ax_fill_betweenx(x1, x2, *, ax=None, **kwargs):

    ax = cast(Axes, ax)

    this_ylim = ax.get_ylim()
    ax.fill_betweenx(this_ylim, [x1, x1], [x2, x2], **kwargs)
    ax.set_ylim(this_ylim)


@auto_setup_ax
def plot_line(*,
              x=None,
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
    ax = cast(Axes, ax)

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


@auto_setup_ax
def plot_log_line(*,
                  p1=None,
                  p2=None,
                  k=None,
                  b=None,
                  ax=None,
                  range=None,
                  N=1000,
                  is_input_log=True,
                  **kwargs):
    ax = cast(Axes, ax)

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
