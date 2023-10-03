import itertools
import re
from typing import Union, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .plot_methods import plot_contour, plot_trend, plot_corner, plot_scatter, plot_heatmap
from .utils import string_to_list, is_string_or_list_of_string


class BasicDataset:

    OP_MAP = {'log10': np.log10, 'square': np.square}
    OP_MAP_LABEL = {'log10': r'$\log$', 'square': ''}

    def __init__(self, data, names=None, labels=None) -> None:
        # TODO: ranges
        # TODO: support labels as a dict = {name: label}

        # If data is pandas DF, convert it to numpy array
        # sourcery skip: merge-else-if-into-elif
        if isinstance(data, pd.DataFrame):
            if names is None:
                names = data.columns
            data = data.to_numpy()
        else:
            if names is None:
                names = [f'x{i}' for i in range(data.shape[1])]

        if labels is None:
            labels = names

        for i, label in enumerate(labels):
            if label is None:
                labels[i] = names[i]

        self.data = np.asarray(data)
        self.names = np.asarray(names)
        self.labels = np.asarray(labels)

        # if data, names, labels have same length
        len_data = self.data.shape[1]
        len_names = self.names.shape[0]
        len_labels = self.labels.shape[0]

        if len_data != len_names:
            raise ValueError('data and names have different length')
        if len_data != len_labels:
            raise ValueError('data and labels have different length')

    def __getitem__(self, key) -> np.ndarray:
        '''
        Get the data by index or name.
        '''

        # support log10@x here

        if isinstance(key, tuple):
            if len(key) != 2:
                raise ValueError('key should be a tuple of length 2')
            if is_string_or_list_of_string(key[0]):
                raise ValueError('key[0] can not be string or list of string')
            if is_string_or_list_of_string(key[1]):
                k1_idx, k2 = key
                names_list = list(self.names)
                if isinstance(k2, str):
                    k2_idx: Union[int, List[int]] = names_list.index(k2)
                else:
                    k2_idx = [names_list.index(this_k2) for this_k2 in k2]
            else:
                k1_idx, k2_idx = key
            return self.data[k1_idx, k2_idx]
        elif is_string_or_list_of_string(key):
            names_list = list(self.names)
            if isinstance(key, str):
                key_idx: Union[int, List[int]] = names_list.index(key)
            else:
                key_idx = [names_list.index(this_k) for this_k in key]
            return self.data[:, key_idx]
        else:
            return self.data[key]

    def __setitem__(self, keys, values) -> None:

        # TODO: reassign existed columns
        self.add_col(values, keys, keys)

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        summary_string = "" + 'Dataset summary:\n'
        summary_string += f'  Data shape: {str(self.data.shape)}' + '\n'
        summary_string += f'  Names: {str(self.names)}' + '\n'
        summary_string += f'  Labels: {str(self.labels)}' + '\n'
        return summary_string

    def to_pandas(self):
        return pd.DataFrame(self.data, columns=self.names)

    def summary(self, stats_info=False) -> None:
        print(self.__str__())
        if stats_info:
            print(self.to_pandas().describe())

    def add_col(self, new_cols, new_names, new_labels) -> None:

        new_names = string_to_list(new_names)
        new_labels = string_to_list(new_labels)

        for name in new_names:
            if name in self.names:
                raise ValueError(f'{name} already exists in the dataset')

        new_cols = np.asarray(new_cols)
        if new_cols.ndim == 1:
            new_cols = new_cols[:, np.newaxis]

        self.data = np.hstack((self.data, new_cols))
        self.names = np.asarray(list(self.names) + list(new_names))
        self.labels = np.asarray(list(self.labels) + list(new_labels))

    def add_row(self, new_rows) -> None:
        self.data = np.vstack((self.data, new_rows))

    def del_col(self, key) -> None:
        '''
        deleta the data by index or name.
        '''
        if is_string_or_list_of_string(key):
            names_list = list(self.names)
            if isinstance(key, str):
                key_idx: Union[int, List[int]] = names_list.index(key)
            else:
                key_idx = [names_list.index(this_k) for this_k in key]
            self._del_col(key_idx)
        else:
            self._del_col(key)

    def _del_col(self, key):
        self.data = np.delete(self.data, key, axis=1)
        self.names = np.delete(self.names, key, axis=0)
        self.labels = np.delete(self.labels, key, axis=0)

    def del_row(self, ncol) -> None:
        self.data = np.delete(self.data, ncol, axis=0)

    def get_data_by_name(self, name):
        # sourcery skip: remove-unnecessary-else, swap-if-else-branches
        if name is None:
            return None
        elif '@' in name:
            op, name = name.split('@')
            return self.OP_MAP[op](self[name])
        else:
            return self[name]

    def get_label_by_name(self, name):
        if name is None:
            return None
        elif '@' in name:
            op, name = name.split('@')
            return self.OP_MAP_LABEL[op] + self.labels[self.names == name][0]
        else:
            return self.labels[self.names == name][0]

    def get_subsample(self, subsample):  # sourcery skip: lift-return-into-if

        if subsample is None:
            _subsample = slice(None)
        elif isinstance(subsample, str):
            _subsample = self.string_to_subsample(subsample)
        else:
            _subsample = subsample

        return _subsample

    def string_to_subsample(self, string):
        # sourcery skip: lift-return-into-if, remove-unnecessary-else

        if is_inequality(string):
            _subsample = self.inequality_to_subsample(string)
        else:
            names_list = list(self.names)
            subsample_idx = names_list.index(string)
            _subsample = self.data[:, subsample_idx].astype(bool)
        return _subsample

    def inequality_to_subsample(self, inequality_string, debug=False):
        '''
        Return the subsample according to the inequality string.
        '''
        # TODO: support & and |
        inequality_list = parse_inequality(inequality_string)
        names_list = list(self.names)
        subsample = np.ones(self.data.shape[0]).astype(bool)

        op_list = ['<=', '>=', '<', '>']
        for i, string in enumerate(inequality_list[2:]):
            if string not in op_list:
                this_inequality = inequality_list[i:i + 3]
                for j in range(len(this_inequality)):
                    if this_inequality[j] in names_list:
                        this_inequality[j] = f"self['{this_inequality[j]}']"

                command = "".join(this_inequality)
                if debug:
                    print(this_inequality)

                subsample = subsample & eval(command)

        return subsample


class Dataset(BasicDataset):

    # TODO: histogram, scatter, etc.
    # TODO: heatmap
    # TODO: control 1D/2D

    def __init__(self, data, names=None, labels=None) -> None:

        super().__init__(data, names=names, labels=labels)

        self.method_mapping = {
            'trend': self._trend,
            'contour': self._contour,
            'scatter': self._scatter,
            'heatmap': self._heatmap
        }

    def _trend(self,
               x_name,
               y_name,
               ax,
               subsample=None,
               xlabel=None,
               ylabel=None,
               title=None,
               xlim=None,
               ylim=None,
               **kwargs):

        x = self.get_data_by_name(x_name)
        y = self.get_data_by_name(y_name)
        _subsample = self.get_subsample(subsample)
        weights = kwargs.pop('weights', None)
        weights = self.get_data_by_name(weights) if isinstance(
            weights, str) else weights
        _weights = weights[_subsample] if weights is not None else None
        plot_trend(x[_subsample],
                   y[_subsample],
                   ax=ax,
                   weights=_weights,
                   **kwargs)
        self._set_ax_prperties(ax, x_name, y_name, xlabel, ylabel, title, xlim,
                               ylim)
        ax.legend()

    def _heatmap(self,
                 x_name,
                 y_name,
                 ax,
                 z_name=None,
                 subsample=None,
                 xlabel=None,
                 ylabel=None,
                 title=None,
                 xlim=None,
                 ylim=None,
                 **kwargs):

        x = self.get_data_by_name(x_name)
        y = self.get_data_by_name(y_name)

        if z_name is None:
            z_name = np.ones_like(x)
            print("z_name is not specified, use np.ones_like(x) instead")
            print("I think you'd like to specify z_name")

        z = self.get_data_by_name(z_name)

        _subsample = self.get_subsample(subsample)
        weights = kwargs.pop('weights', None)
        weights = self.get_data_by_name(weights) if isinstance(
            weights, str) else weights
        _weights = weights[_subsample] if weights is not None else None
        plot_heatmap(x[_subsample],
                     y[_subsample],
                     z[_subsample],
                     ax=ax,
                     weights=_weights,
                     **kwargs)

        self._set_ax_prperties(ax, x_name, y_name, xlabel, ylabel, title, xlim,
                               ylim)

    def _contour(self,
                 x_name,
                 y_name,
                 ax,
                 subsample=None,
                 xlabel=None,
                 ylabel=None,
                 title=None,
                 xlim=None,
                 ylim=None,
                 **kwargs):
        '''
        xlabel:
            If False, do not set xlabel
            If None, set from self.labels
            If string, set as xlabel
        
        ylabel:
            If False, do not set ylabel
            If None, set from self.labels
            If string, set as ylabel

        title:
            If None or False, do not set title
            If string, set as title

        xlim:
            If None, do not set xlim
            If list, set as xlim

        ylim:
            If None, do not set ylim
            If list, set as ylim

        '''

        x = self.get_data_by_name(x_name)
        y = self.get_data_by_name(y_name)
        _subsample = self.get_subsample(subsample)
        weights = kwargs.pop('weights', None)
        weights = self.get_data_by_name(weights) if isinstance(
            weights, str) else weights
        _weights = weights[_subsample] if weights is not None else None
        plot_contour(x[_subsample],
                     y[_subsample],
                     ax=ax,
                     weights=_weights,
                     **kwargs)

        self._set_ax_prperties(ax, x_name, y_name, xlabel, ylabel, title, xlim,
                               ylim)

    def _set_ax_prperties(self, ax, x_name, y_name, xlabel, ylabel, title,
                          xlim, ylim):
        if (title is not False) and (title is not None):
            ax.set_title(title)

        if xlabel is not False:
            if xlabel is None:
                xlabel = self.get_label_by_name(x_name)
            ax.set_xlabel(xlabel)

        if ylabel is not False:
            if ylabel is None:
                ylabel = self.get_label_by_name(y_name)
            ax.set_ylabel(ylabel)

        if xlim is not None:
            ax.set_xlim(xlim)

        if ylim is not None:
            ax.set_ylim(ylim)

    def _scatter(self,
                 x_name,
                 y_name,
                 ax,
                 subsample=None,
                 xlabel=None,
                 ylabel=None,
                 title=None,
                 xlim=None,
                 ylim=None,
                 **kwargs):

        x = self.get_data_by_name(x_name)
        y = self.get_data_by_name(y_name)
        _subsample = self.get_subsample(subsample)
        weights = kwargs.pop('weights', None)
        weights = self.get_data_by_name(weights) if isinstance(
            weights, str) else weights
        _weights = weights[_subsample] if weights is not None else None
        plot_scatter(x[_subsample],
                     y[_subsample],
                     ax=ax,
                     weights=_weights,
                     **kwargs)
        self._set_ax_prperties(ax, x_name, y_name, xlabel, ylabel, title, xlim,
                               ylim)
        if kwargs.get('label', None) is not None:
            ax.legend()

    def plot_xygeneral(self,
                       kind,
                       x_names,
                       y_names,
                       axes=None,
                       subplots_kwargs=None,
                       **kwargs):

        # TODO: bin by the third variable
        # TODO: No broadcast option

        x_names = string_to_list(x_names)
        y_names = string_to_list(y_names)

        n1 = len(x_names)
        n2 = len(y_names)

        if (n1 > 1) and (n2 > 1):
            scatter_type = 'xy'
        elif (n1 > 1) and (n2 == 1):
            scatter_type = 'x'
        elif (n1 == 1) and (n2 > 1):
            scatter_type = 'y'
        elif (n1 == 1) and (n2 == 1):
            scatter_type = 'single'

        if subplots_kwargs is None:
            subplots_kwargs = {}

        if axes is None:
            if scatter_type == 'xy':
                fig, axes = auto_subplots(n1, n2, **subplots_kwargs)
            else:
                fig, axes = auto_subplots(n1 * n2, **subplots_kwargs)

        # If axes is a single ax, convert it to an array
        if not hasattr(axes, '__iter__'):
            axes = np.array([axes])

        # find fig by axes
        fig = axes.flatten()[0].get_figure()

        same_key = {}
        each_key = {}
        for key in kwargs:
            # is end of
            if key.endswith('_each'):
                key_single = key[:-5]
                each_key[key_single] = kwargs[key]
                # If it is 1D list, convert it to 2D list
                '''
                Note that this code can not solve all problems
                Corner case: range = [[1, 2], [1, 2]]
                No plan to deal with such corner case
                If anyone has a good idea, please let me know
                '''
                if not isinstance(each_key[key_single][0], list):
                    each_key[key_single] = [each_key[key_single]]
            else:
                same_key[key] = kwargs[key]

        # i, vertical, y; j, horizontal, x
        for (i, j), ax in zip(itertools.product(range(n2), range(n1)),
                              axes.flatten()):
            this_kwargs = same_key.copy()

            for key in each_key:
                if scatter_type == 'xy':
                    this_kwargs[key] = each_key[key][i][j]
                elif scatter_type in ['x', 'single']:
                    this_kwargs[key] = each_key[key][0][j]
                elif scatter_type == 'y':
                    this_kwargs[key] = each_key[key][0][i]

            self.method_mapping[kind](x_names[j], y_names[i], ax,
                                      **this_kwargs)
        
        return fig, axes

    def trend(self,
              x_names,
              y_names,
              axes=None,
              subplots_kwargs=None,
              **kwargs):

        return self.plot_xygeneral('trend',
                            x_names,
                            y_names,
                            axes=axes,
                            subplots_kwargs=subplots_kwargs,
                            **kwargs)

    def contour(self,
                x_names,
                y_names,
                axes=None,
                subplots_kwargs=None,
                **kwargs):

        return self.plot_xygeneral('contour',
                            x_names,
                            y_names,
                            axes=axes,
                            subplots_kwargs=subplots_kwargs,
                            **kwargs)

    def corner(self, names=None, axes=None, **kwargs):
        if names is None:
            names = self.names

        if 'labels' not in kwargs:
            kwargs['labels'] = [self.get_label_by_name(name) for name in names]

        if axes is not None:
            axes = np.atleast_1d(axes)
            fig = axes.flatten()[0].get_figure()
        else:
            fig = None

        xs = np.array([self.get_data_by_name(name) for name in names]).T
        return plot_corner(xs, fig=fig, **kwargs)

    def scatter(self,
                x_names,
                y_names,
                axes=None,
                subplots_kwargs=None,
                **kwargs):

        return self.plot_xygeneral('scatter',
                            x_names,
                            y_names,
                            axes=axes,
                            subplots_kwargs=subplots_kwargs,
                            **kwargs)

    def heatmap(self,
                x_names,
                y_names,
                axes=None,
                subplots_kwargs=None,
                **kwargs):
        return self.plot_xygeneral('heatmap',
                            x_names,
                            y_names,
                            axes=axes,
                            subplots_kwargs=subplots_kwargs,
                            **kwargs)


def auto_subplots(n1, n2=None, figshape=None, figsize=None, dpi=400):
    if figshape is None:
        if n2 is None:
            figshape = (int(np.ceil(np.sqrt(n1))), int(np.ceil(np.sqrt(n1))))
        else:
            figshape = (n2, n1)  # vertical, horizontal
    if figsize is None:
        figsize = (figshape[1] * 4, figshape[0] * 4)
    fig, axes = plt.subplots(figshape[0],
                             figshape[1],
                             figsize=figsize,
                             dpi=400)
    if not hasattr(axes, '__iter__'):
        axes = np.array([axes])
    return fig, axes


def parse_inequality(inequaliyt_string):
    return re.split(r'(<=|>=|<|>)', inequaliyt_string.replace(" ", ""))


def is_inequality(string):
    return re.search(r'(<=|>=|<|>)', string) is not None
