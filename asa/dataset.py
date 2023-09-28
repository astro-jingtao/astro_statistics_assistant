import itertools
import re
from typing import Union, List
import numpy as np
import matplotlib.pyplot as plt
from .plot_methods import plot_contour, plot_trend, plot_corner
from .utils import string_to_list, is_string_or_list_of_string

class Dataset:

    # TODO: histogram, scatter, etc.
    # TODO: heatmap

    OP_MAP = {'log10': np.log10}

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
        if '@' in name:
            op, name = name.split('@')
            return self.OP_MAP[op](self[name])
        else:
            return self[name]

    def _trend(self, x_name, y_name, ax, subsample=None, **kwargs):

        # TODO: label, etc.

        x = self.get_data_by_name(x_name)
        y = self.get_data_by_name(y_name)
        _subsample = self.get_subsample(subsample)
        plot_trend(x[_subsample], y[_subsample], ax=ax, **kwargs)
        ax.set_xlabel(x_name)
        ax.set_ylabel(y_name)
        ax.legend()

    def _contour(self, x_name, y_name, ax, subsample=None, **kwargs):

        # TODO: labels, titles, ranges, etc.

        x = self.get_data_by_name(x_name)
        y = self.get_data_by_name(y_name)
        _subsample = self.get_subsample(subsample)
        plot_contour(x[_subsample], y[_subsample], ax=ax, **kwargs)
        ax.set_xlabel(x_name)
        ax.set_ylabel(y_name)


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

    def plot_xygeneral(self,
                       kind,
                       x_names,
                       y_names,
                       subsample=None,
                       axes=None,
                       subplots_kwargs=None,
                       **kwargs):

        # TODO: contour plot bin by the third variable
        # TODO: subsample deal with weight
        # TODO: can set color, weight ... by name

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
                # if is 1D list, convert it to 2D list
                if not isinstance(each_key[key_single][0], list):
                    each_key[key_single] = [each_key[key_single]]
            else:
                same_key[key] = kwargs[key]

        # i, vertical, y; j, horizontal, x
        for (j, i), ax in zip(itertools.product(range(n1), range(n2)),
                              axes.T.flatten()):
            this_kwargs = same_key.copy()

            for key in each_key:
                this_kwargs[key] = each_key[key][j][i]
            if scatter_type == 'xy':
                self.method_mapping[kind](x_names[j],
                                          y_names[i],
                                          ax,
                                          subsample=subsample,
                                          **this_kwargs)
            elif scatter_type in ['x', 'single']:
                self.method_mapping[kind](x_names[j],
                                          y_names[0],
                                          ax,
                                          subsample=subsample,
                                          **this_kwargs)
            elif scatter_type == 'y':
                self.method_mapping[kind](x_names[0],
                                          y_names[j],
                                          ax,
                                          subsample=subsample,
                                          **this_kwargs)

    def trend(self,
              x_names,
              y_names,
              subsample=None,
              axes=None,
              subplots_kwargs=None,
              **kwargs):

        self.plot_xygeneral('trend',
                            x_names,
                            y_names,
                            subsample=subsample,
                            axes=axes,
                            subplots_kwargs=subplots_kwargs,
                            **kwargs)

    def contour(self,
                x_names,
                y_names,
                subsample=None,
                axes=None,
                subplots_kwargs=None,
                **kwargs):

        self.plot_xygeneral('contour',
                            x_names,
                            y_names,
                            subsample=subsample,
                            axes=axes,
                            subplots_kwargs=subplots_kwargs,
                            **kwargs)
        
    def corner(self, names=None, axes=None, **kwargs):
        '''
        Plot the corner plot.
        '''

        # TODO: auto set labels
        xs = np.array([self.get_data_by_name(name) for name in names]).T
        return plot_corner(xs, **kwargs)



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
