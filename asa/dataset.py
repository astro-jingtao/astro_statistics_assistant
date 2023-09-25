import numpy as np
import matplotlib.pyplot as plt
from .plot_contour import plot_contour
from .plot_trend import plot_trend
from .utils import string_to_list, is_string_or_list_of_string


class Dataset:

    def __init__(self, data, names, labels) -> None:

        # TODO: ranges

        self.data = np.asarray(data)
        self.names = np.asarray(names)
        self.labels = np.asarray(labels)
        self.method_mapping = {'trend': self._trend, 'contour': self._contour}

    def __getitem__(self, key) -> np.ndarray:
        '''
        Get the data by index or name.
        '''

        if isinstance(key, tuple):
            if len(key) != 2:
                raise ValueError('key should be a tuple of length 2')
            if is_string_or_list_of_string(key[0]):
                raise ValueError('key[0] can not be string or list of string')
            if is_string_or_list_of_string(key[1]):
                k1_idx, k2 = key
                names_list = list(self.names)
                if isinstance(k2, str):
                    k2_idx = names_list.index(k2)
                else:
                    k2_idx = [names_list.index(this_k2) for this_k2 in k2]
            else:
                k1_idx, k2_idx = key
            return self.data[k1_idx, k2_idx]
        elif is_string_or_list_of_string(key):
            names_list = list(self.names)
            if isinstance(key, str):
                key_idx = names_list.index(key)
            else:
                key_idx = [names_list.index(this_k) for this_k in key]
            return self.data[:, key_idx]
        else:
            return self.data[key]

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        summary_string = "" + 'Dataset summary:\n'
        summary_string += f'  Data shape: {str(self.data.shape)}' + '\n'
        summary_string += f'  Names: {str(self.names)}' + '\n'
        summary_string += f'  Labels: {str(self.labels)}' + '\n'
        return summary_string

    def summary(self) -> None:

        # TODO: math summary?

        print(self.__str__())

    def add_col(self, new_cols, new_names, new_labels) -> None:

        new_cols = np.asarray(new_cols)
        if new_cols.ndim == 1:
            new_cols = new_cols[:, np.newaxis]
        new_names = string_to_list(new_names)
        new_labels = string_to_list(new_labels)

        self.data = np.hstack((self.data, new_cols))
        self.names = np.asarray(list(self.names) + list(new_names))
        self.labels = np.asarray(list(self.labels) + list(new_labels))

    def add_row(self, new_rows) -> None:
        self.data = np.vstack((self.data, new_rows))

    def _trend(self, x_name, y_name, ax, subsample=None, **kwargs):

        # TODO: label, etc.

        names_list = list(self.names)
        x_idx = names_list.index(x_name)
        y_idx = names_list.index(y_name)
        if subsample is None:
            plot_trend(self.data[:, x_idx],
                       self.data[:, y_idx],
                       ax=ax,
                       **kwargs)
        else:
            plot_trend(self.data[subsample, x_idx],
                       self.data[subsample, y_idx],
                       ax=ax,
                       **kwargs)
        ax.set_xlabel(x_name)
        ax.set_ylabel(y_name)
        ax.legend()

    def _contour(self, x_name, y_name, ax, subsample=None, **kwargs):

        # TODO: labels, titles, ranges, etc.

        names_list = list(self.names)
        x_idx = names_list.index(x_name)
        y_idx = names_list.index(y_name)
        if subsample is None:
            plot_contour(self.data[:, x_idx],
                         self.data[:, y_idx],
                         ax=ax,
                         **kwargs)
        else:
            plot_contour(self.data[subsample, x_idx],
                         self.data[subsample, y_idx],
                         ax=ax,
                         **kwargs)
        ax.set_xlabel(x_name)
        ax.set_ylabel(y_name)

    def plot_xygeneral(self,
                       kind,
                       x_name,
                       y_names,
                       subsample=None,
                       axes=None,
                       subplots_kwargs=None,
                       **kwargs):

        # TODO: contour plot bin by the third variable
        # TODO: data transformation

        y_names = string_to_list(y_names)

        if subplots_kwargs is None:
            subplots_kwargs = {}

        if axes is None:
            _, axes = auto_subplots(len(y_names), **subplots_kwargs)

        same_key = {}
        each_key = {}
        for key in kwargs:
            # is end of
            if key.endswith('_each'):
                key_single = key[:-5]
                each_key[key_single] = kwargs[key]
            else:
                same_key[key] = kwargs[key]

        for i, ax in enumerate(axes.flatten()):
            this_kwargs = same_key.copy()
            for key in each_key:
                this_kwargs[key] = each_key[key][i]
            self.method_mapping[kind](x_name,
                                      y_names[i],
                                      ax,
                                      subsample=subsample,
                                      **this_kwargs)

    def trend(self,
              x_name,
              y_names,
              subsample=None,
              axes=None,
              subplots_kwargs=None,
              **kwargs):

        self.plot_xygeneral('trend',
                            x_name,
                            y_names,
                            subsample=subsample,
                            axes=axes,
                            subplots_kwargs=subplots_kwargs,
                            **kwargs)

    def contour(self,
                x_name,
                y_names,
                subsample=None,
                axes=None,
                subplots_kwargs=None,
                **kwargs):

        self.plot_xygeneral('contour',
                            x_name,
                            y_names,
                            subsample=subsample,
                            axes=axes,
                            subplots_kwargs=subplots_kwargs,
                            **kwargs)


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
