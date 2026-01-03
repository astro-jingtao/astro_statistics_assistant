import functools
from typing import Final

import matplotlib.gridspec as gridspec
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import numpy as np
from asa.utils import (all_asarray, any_empty, remove_bad,
                       ensure_parameter_spec, is_empty)


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


def get_subplots(nrows,
                 ncols,
                 squeeze=True,
                 left=None,
                 right=None,
                 bottom=None,
                 top=None,
                 wspaces=None,
                 hspaces=None,
                 height_ratios=None,
                 width_ratios=None,
                 **fig_kwargs):

    if wspaces is None:
        wspaces = [0.2] * (ncols - 1)
    elif isinstance(wspaces, float) or isinstance(wspaces, int):
        wspaces = [wspaces] * (ncols - 1)
    if hspaces is None:
        hspaces = [0.2] * (nrows - 1)
    elif isinstance(hspaces, float) or isinstance(hspaces, int):
        hspaces = [hspaces] * (nrows - 1)

    if width_ratios is None:
        width_ratios = [1] * ncols
    if height_ratios is None:
        height_ratios = [1] * nrows

    gs_width_ratios = []
    for panel, space in zip(width_ratios[:-1], wspaces):
        gs_width_ratios += [panel, space]
    gs_width_ratios.append(width_ratios[-1])

    gs_height_ratios = []
    for panel, space in zip(height_ratios[:-1], hspaces):
        gs_height_ratios += [panel, space]
    gs_height_ratios.append(height_ratios[-1])

    gs = gridspec.GridSpec(nrows * 2 - 1,
                           ncols * 2 - 1,
                           left=left,
                           right=right,
                           bottom=bottom,
                           top=top,
                           wspace=0,
                           hspace=0,
                           width_ratios=gs_width_ratios,
                           height_ratios=gs_height_ratios)
    fig = plt.figure(**fig_kwargs)

    axes = np.empty((nrows, ncols), dtype=object)
    for i in range(nrows):
        for j in range(ncols):
            axes[i, j] = fig.add_subplot(gs[2 * i, 2 * j])

    if squeeze:
        return fig, np.squeeze(axes)
    else:
        return fig, axes


def split_ax(ax,
             nrows,
             ncols,
             ax_off=True,
             wspace=None,
             hspace=None,
             height_ratios=None,
             width_ratios=None):

    sub_sub_gridspec = gridspec.GridSpecFromSubplotSpec(
        nrows,
        ncols,
        subplot_spec=ax.get_subplotspec(),
        wspace=wspace,
        hspace=hspace,
        height_ratios=height_ratios,
        width_ratios=width_ratios)

    axes = []

    for i in range(nrows):
        for j in range(ncols):
            axes.append(ax.figure.add_subplot(sub_sub_gridspec[i, j]))

    if ax_off:
        ax.axis('off')

    return np.squeeze(np.array(axes).reshape(nrows, ncols))


COLOR_CYCLE: Final = plt.rcParams['axes.prop_cycle'].by_key()['color']


class ColorCycler:

    def __init__(self):
        # self.color_index = 0
        # self.current_ax = None
        self.record = {}

    def next(self, ax=None):
        if ax is None:
            ax = plt.gca()
        if ax not in self.record:
            self.record[ax] = 0
        color = COLOR_CYCLE[self.record[ax]]
        self.record[ax] = (self.record[ax] + 1) % len(COLOR_CYCLE)
        return color


class Jitter:

    def __init__(self):
        ...

    def jit(self, x):
        ...


class GaussianJitter(Jitter):

    def __init__(self, scale=0.1):
        self.scale = scale
        super().__init__()

    def jitter(self, x):
        return x + np.random.normal(0, self.scale, size=x.size)


class UniformJitter(Jitter):

    def __init__(self, scale=0.1):
        self.scale = scale
        super().__init__()

    def jitter(self, x):
        return x + np.random.uniform(
            -self.scale / 2, self.scale / 2, size=x.size)


class FixedJitter(Jitter):

    def __init__(self, offset=0.1):
        self.offset = offset
        super().__init__()

    def jitter(self, x):
        return x + self.offset


def jitter_data(x, jitter):
    if jitter is None:
        return x
    if isinstance(jitter, tuple):
        if jitter[0] == 'gaussian':
            jitter = GaussianJitter(scale=jitter[1])
        elif jitter[0] == 'uniform':
            jitter = UniformJitter(scale=jitter[1])
        elif jitter[0] == 'fixed':
            jitter = FixedJitter(offset=jitter[1])
        else:
            raise ValueError(f'Invalid jitter type: {jitter[0]}')
    return jitter.jitter(x)


def auto_setup_ax(func):
    """
    Decorator: Automatically set 'ax' to plt.gca() if it's None.
    Required signature: def func(..., ax=None)
    """
    # Check signature during decoration time (when the script starts)
    ensure_parameter_spec(func, 'ax', None)

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        ax = kwargs.get('ax')

        if ax is None:
            ax = plt.gca()
        # is an array of axes with only one element
        elif isinstance(ax, np.ndarray) and ax.size == 1:
            print(
                "Warning: 'ax' is an array with only one element, unwrapping it to single Axes instance. If 'ax' will be returned by the function, it will be an unwrapped Axes instance rather than original array."
            )
            ax = ax.item()

        if not isinstance(ax, Axes):
            raise ValueError(
                f"'ax' should be an instance of matplotlib.axes.Axes, an array of only one such element, or None, got {type(ax)}."
            )

        kwargs['ax'] = ax

        return func(*args, **kwargs)

    return wrapper


def prepare_data(*args, arg_names=None, to_transpose=None):
    """
    all_asarray + remove_bad + check empty
    """
    args = all_asarray(args)
    args = remove_bad(args, to_transpose=to_transpose)

    if arg_names is None:
        if any_empty(args):
            raise ValueError('Some data are empty after removing bad values.')
    elif len(arg_names) != len(args):
        raise ValueError(
            f'Number of {arg_names} does not match the number of data.')
    else:
        empty_lst = []
        for arg, name in zip(args, arg_names):
            if (arg is not None) and is_empty(arg):
                empty_lst.append(name)
        if empty_lst:
            raise ValueError(
                f'Some {arg_names} are empty after removing bad values: {empty_lst}.'
            )

    if len(args) == 1:
        args = args[0]

    return args
