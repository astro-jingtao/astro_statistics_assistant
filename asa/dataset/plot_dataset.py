import itertools

import matplotlib.pyplot as plt
import numpy as np

from ..plot_methods import (plot_contour, plot_corner, plot_heatmap,
                            plot_sample_to_point, plot_scatter, plot_trend)

from ..utils import (flag_bad, string_to_list, all_subsample)

from .basic_dataset import BasicDataset


class PlotDataset(BasicDataset):

    # -- Note -- that all values passed to plot_xxx should be numpy array, not series

    # TODO: histogram
    # TODO: control 1D/2D

    def __init__(self,
                 data,
                 names=None,
                 labels=None,
                 ranges=None,
                 unit_labels=None,
                 units=None,
                 snr_postfix='snr',
                 err_postfix='err') -> None:

        super().__init__(data,
                         names=names,
                         labels=labels,
                         ranges=ranges,
                         unit_labels=unit_labels,
                         units=units,
                         snr_postfix=snr_postfix,
                         err_postfix=err_postfix)

        self.method_mapping = {
            'trend': self._trend,
            'contour': self._contour,
            'scatter': self._scatter,
            'heatmap': self._heatmap,
            'sample_to_point': self._sample_to_point
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
               legend_kwargs=None,
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
        self._set_ax_properties(ax, x_name, y_name, xlabel, ylabel, title,
                                xlim, ylim)

        for arg_key in ['errorbar_kwargs', 'fbetween_kwargs', 'plot_kwargs']:
            if arg_key in kwargs:
                if 'label' in kwargs[arg_key]:
                    if legend_kwargs is None:
                        legend_kwargs = {}
                    ax.legend(**legend_kwargs)
                    break


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
            z = np.ones_like(x)
            print("z_name is not specified, use z = np.ones_like(x)")
            print("I think you'd like to specify z_name")
        else:
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

        if title is None:
            title = self.get_label_by_name(z_name)

        self._set_ax_properties(ax, x_name, y_name, xlabel, ylabel, title,
                                xlim, ylim)

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

        if 'range' not in kwargs:
            kwargs['range'] = self._get_default_range(x_name, y_name)

        plot_contour(x[_subsample],
                     y[_subsample],
                     ax=ax,
                     weights=_weights,
                     **kwargs)

        self._set_ax_properties(ax, x_name, y_name, xlabel, ylabel, title,
                                xlim, ylim)

    def _get_default_range(self, x_name, y_name):
        xrange = self.get_range_by_name(x_name)
        if xrange is None:
            x = self.get_data_by_name(x_name)
            x = x[~flag_bad(x)]
            xrange = [x.min(), x.max()]
        yrange = self.get_range_by_name(y_name)
        if yrange is None:
            y = self.get_data_by_name(y_name)
            y = y[~flag_bad(y)]
            yrange = [y.min(), y.max()]
        return [xrange, yrange]

    # TODO: set font size
    def _set_ax_properties(self, ax, x_name, y_name, xlabel, ylabel, title,
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

        if xlim is not False:
            if xlim is None:
                xlim = self.get_range_by_name(x_name)
            # if x_name is not in self.ranges, xlim will also be None
            if xlim is not None:
                ax.set_xlim(xlim)

        if ylim is not False:
            if ylim is None:
                ylim = self.get_range_by_name(y_name)
            # if y_name is not in self.ranges, ylim will also be None
            if ylim is not None:
                ax.set_ylim(ylim)

    def _scatter(self,
                 x_name,
                 y_name,
                 ax,
                 z_name=None,
                 xerr_name=None,
                 yerr_name=None,
                 subsample=None,
                 xlabel=None,
                 ylabel=None,
                 title=None,
                 xlim=None,
                 ylim=None,
                 legend_kwargs=None,
                 **kwargs):

        x = self.get_data_by_name(x_name)
        y = self.get_data_by_name(y_name)
        _z = None if (z_name is None) else self.get_data_by_name(z_name)
        _xerr = None if (xerr_name
                         is None) else self.get_data_by_name(xerr_name)
        _yerr = None if (yerr_name
                         is None) else self.get_data_by_name(yerr_name)
        _subsample = self.get_subsample(subsample)
        weights = kwargs.pop('weights', None)
        weights = self.get_data_by_name(weights) if isinstance(
            weights, str) else weights
        _weights = weights[_subsample] if weights is not None else None

        _x, _y, _z, _xerr, _yerr = all_subsample([x, y, _z, _xerr, _yerr],
                                                 _subsample)

        plot_scatter(_x,
                     _y,
                     z=_z,
                     xerr=_xerr,
                     yerr=_yerr,
                     ax=ax,
                     weights=_weights,
                     **kwargs)
        self._set_ax_properties(ax, x_name, y_name, xlabel, ylabel, title,
                                xlim, ylim)
        if kwargs.get('label', None) is not None:
            if legend_kwargs is None:
                legend_kwargs = {}
            ax.legend(**legend_kwargs)

    def _sample_to_point(self,
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
        plot_sample_to_point(x[_subsample],
                             y[_subsample],
                             ax=ax,
                             weights=_weights,
                             **kwargs)

        self._set_ax_properties(ax, x_name, y_name, xlabel, ylabel, title,
                                xlim, ylim)

    def plot_xygeneral_no_broadcast(self,
                                    kind,
                                    x_names,
                                    y_names,
                                    axes=None,
                                    subplots_kwargs=None,
                                    **kwargs):

        # TODO: return all extra returns by each method

        x_names = string_to_list(x_names)
        y_names = string_to_list(y_names)

        if len(x_names) != len(y_names):
            raise ValueError('x_names and y_names have different length')

        if subplots_kwargs is None:
            subplots_kwargs = {}

        if axes is None:
            fig, axes = auto_subplots(len(x_names), **subplots_kwargs)

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
            else:
                same_key[key] = kwargs[key]

        for i in range(len(x_names)):
            ax = axes.flat[i]
            this_kwargs = same_key.copy()
            for key in each_key:
                this_kwargs[key] = each_key[key][i]
            self.method_mapping[kind](x_names[i], y_names[i], ax,
                                      **this_kwargs)

        return fig, axes

    def plot_xygeneral(self,
                       kind,
                       x_names,
                       y_names,
                       axes=None,
                       subplots_kwargs=None,
                       **kwargs):

        # TODO: bin by the third variable
        # TODO: return all extra returns by each method

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
              broadcast=True,
              axes=None,
              subplots_kwargs=None,
              **kwargs):

        if broadcast:
            return self.plot_xygeneral('trend',
                                       x_names,
                                       y_names,
                                       axes=axes,
                                       subplots_kwargs=subplots_kwargs,
                                       **kwargs)

        else:
            return self.plot_xygeneral_no_broadcast(
                'trend',
                x_names,
                y_names,
                axes=axes,
                subplots_kwargs=subplots_kwargs,
                **kwargs)

    def contour(self,
                x_names,
                y_names,
                broadcast=True,
                axes=None,
                subplots_kwargs=None,
                **kwargs):
        """

        Parameters
        ----------
        x_names: str or list of str
            The names of x variables
        
        y_names: str or list of str
            The names of y variables
        
        broadcast: bool
            If True, broadcast the plot to all combinations of x and y.
            If False, plot each combination separately.

        axes: matplotlib.axes.Axes or array of Axes
            The axes to plot on. If None, create new axes

        subplots_kwargs: dict
            The kwargs for auto_subplots

        **kwargs:
            The kwargs for self.plot_xygeneral or self.plot_xygeneral_no_broadcast
        """

        if broadcast:
            return self.plot_xygeneral('contour',
                                       x_names,
                                       y_names,
                                       axes=axes,
                                       subplots_kwargs=subplots_kwargs,
                                       **kwargs)
        else:
            return self.plot_xygeneral_no_broadcast(
                'contour',
                x_names,
                y_names,
                axes=axes,
                subplots_kwargs=subplots_kwargs,
                **kwargs)

    def corner(self, names=None, axes=None, **kwargs):
        # TODO: doc string
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
                broadcast=True,
                axes=None,
                subplots_kwargs=None,
                **kwargs):

        if broadcast:
            return self.plot_xygeneral('scatter',
                                       x_names,
                                       y_names,
                                       axes=axes,
                                       subplots_kwargs=subplots_kwargs,
                                       **kwargs)
        else:
            return self.plot_xygeneral_no_broadcast(
                'scatter',
                x_names,
                y_names,
                axes=axes,
                subplots_kwargs=subplots_kwargs,
                **kwargs)

    def heatmap(self,
                x_names,
                y_names,
                broadcast=True,
                axes=None,
                subplots_kwargs=None,
                **kwargs):

        if broadcast:
            return self.plot_xygeneral('heatmap',
                                       x_names,
                                       y_names,
                                       axes=axes,
                                       subplots_kwargs=subplots_kwargs,
                                       **kwargs)
        else:
            return self.plot_xygeneral_no_broadcast(
                'heatmap',
                x_names,
                y_names,
                axes=axes,
                subplots_kwargs=subplots_kwargs,
                **kwargs)

    def sample_to_point(self,
                        x_names,
                        y_names,
                        broadcast=True,
                        axes=None,
                        subplots_kwargs=None,
                        **kwargs):
        if broadcast:
            return self.plot_xygeneral('sample_to_point',
                                       x_names,
                                       y_names,
                                       axes=axes,
                                       subplots_kwargs=subplots_kwargs,
                                       **kwargs)
        else:
            return self.plot_xygeneral_no_broadcast(
                'sample_to_point',
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
                             dpi=dpi)
    if not hasattr(axes, '__iter__'):
        axes = np.array([axes])
    return fig, axes
