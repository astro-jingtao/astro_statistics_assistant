from typing import Final

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


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
        self.color_index = 0
        self.current_ax = None

    def next(self, ax=None):
        if ax is None:
            ax = plt.gca()
        if ax is not self.current_ax:
            self.color_index = 0
            self.current_ax = ax
        color = COLOR_CYCLE[self.color_index]
        self.color_index = (self.color_index + 1) % len(COLOR_CYCLE)
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
