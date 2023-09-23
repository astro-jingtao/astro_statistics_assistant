import numpy as np
import matplotlib.pyplot as plt
from .plot_contour import plot_contour
from .plot_trend import plot_trend


class Dataset:

    def __init__(self, data, names, labels) -> None:
        self.data = data
        self.names = names
        self.labels = labels

    def _trend(self, x_name, y_name, ax, **kwargs):

        # TODO: scatter, etc.

        names_list = list(self.names)
        x_idx = names_list.index(x_name)
        y_idx = names_list.index(y_name)
        plot_trend(self.data[:, x_idx], self.data[:, y_idx], ax=ax, **kwargs)
        ax.set_xlabel(x_name)
        ax.set_ylabel(y_name)

    def trend(self,
                x_name,
                y_names,
                axes=None,
                subplots_kwargs=None,
                **kwargs):

        # TODO: kwargs_each to use different kwargs for each plot

        # if y_names is a string
        if isinstance(y_names, str):
            y_names = [y_names]

        if subplots_kwargs is None:
            subplots_kwargs = {}

        _, axes = auto_subplots(len(y_names), **subplots_kwargs)

        for y_name, ax in zip(y_names, axes.flatten()):
            self._trend(x_name, y_name, ax, **kwargs)


    def _contour(self, x_name, y_name, ax, **kwargs):

        # TODO: labels, tiles, ranges, etc.

        names_list = list(self.names)
        x_idx = names_list.index(x_name)
        y_idx = names_list.index(y_name)
        plot_contour(self.data[:, x_idx], self.data[:, y_idx], ax=ax, **kwargs)
        ax.set_xlabel(x_name)
        ax.set_ylabel(y_name)

    def contour(self,
                x_name,
                y_names,
                axes=None,
                subplots_kwargs=None,
                **kwargs):

        # TODO: kwargs_each to use different kwargs for each plot

        # if y_names is a string
        if isinstance(y_names, str):
            y_names = [y_names]

        if subplots_kwargs is None:
            subplots_kwargs = {}

        _, axes = auto_subplots(len(y_names), **subplots_kwargs)

        for y_name, ax in zip(y_names, axes.flatten()):
            self._contour(x_name, y_name, ax, **kwargs)


def auto_subplots(n, figshape=None, figsize=None, dpi=400):
    if figshape is None:
        figshape = (int(np.ceil(np.sqrt(n))), int(np.ceil(np.sqrt(n))))
    if figsize is None:
        figsize = (figshape[1] * 4, figshape[0] * 4)
    fig, axes = plt.subplots(figshape[0],
                             figshape[1],
                             figsize=figsize,
                             dpi=400)
    if n == 1:
        axes = np.array([axes])
    return fig, axes
