import numpy as np
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
