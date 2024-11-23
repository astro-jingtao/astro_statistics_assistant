import itertools
import warnings

import matplotlib.pyplot as plt
import numpy as np

from ..correlation_methods import get_RF_importance
from ..feature_selection_methods import (search_combination_OLS,
                                         search_combination_RF_cls,
                                         search_combination_RF_reg)
from ..plot_methods import (plot_contour, plot_corner, plot_heatmap,
                            plot_sample_to_point, plot_scatter, plot_trend)
from ..projection_methods import get_LDA_projection
from ..utils import (balance_class, flag_bad, is_bool, is_float, is_int,
                     remove_bad, string_to_list, all_subsample)

from .basic_dataset import BasicDataset

# TODO: DF to AASTeX tabel. Maybe ref to: https://github.com/liuguanfu1120/Excel-to-AASTeX/blob/main/xlsx-to-AAS-table.ipynb


# TODO: split PlotDataset and Dataset
class Dataset(BasicDataset):

    # -- Note -- that all values passed to plot_xxx should be numpy array, not series

    # TODO: histogram
    # TODO: control 1D/2D
    # TODO: inherit the doc string of wrapped methods

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

        if xlim is not None:
            ax.set_xlim(xlim)

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
            ax.legend()

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

    def get_RF_importance_bootstrap(self,
                                    x_names,
                                    y_name,
                                    problem_type=None,
                                    max_sample=None,
                                    subsample=None,
                                    bad_treatment='drop',
                                    auto_balance=False,
                                    check_res=True,
                                    return_more=False,
                                    N_bootstrap=10,
                                    f_bootstrap=0.8,
                                    **kwargs):
        '''
        N_bootstrap: int
            The times of bootstrap
        f_bootstrap: float or int
            If < 1, the fraction of samples used in each bootstrap
            If int, the fraction of samples used in each bootstrap
        '''

        # TODO: return more

        importance_list = []
        test_score_list = []
        for _ in range(N_bootstrap):
            # print(f'Bootstrap {i+1}/{N_bs}')

            this_subsample = self.random_subsample(f_bootstrap,
                                                   input_subsample=subsample,
                                                   as_bool=True)

            importance, test_score = self.get_RF_importance(
                x_names,
                y_name,
                problem_type=problem_type,
                max_sample=max_sample,
                subsample=this_subsample,
                bad_treatment=bad_treatment,
                auto_balance=auto_balance,
                check_res=check_res,
                return_more=False,
                **kwargs)
            importance_list.append(importance)
            test_score_list.append(test_score)

        importance_list = np.array(importance_list)
        test_score_list = np.array(test_score_list)
        return importance_list, test_score_list

    def get_RF_importance(self,
                          x_names,
                          y_name,
                          problem_type=None,
                          max_sample=None,
                          subsample=None,
                          bad_treatment='drop',
                          auto_balance=False,
                          check_res=True,
                          return_more=False,
                          **kwargs):
        # TODO: auto tune hyperparameters
        # TODO: auto get label name
        '''
        problem_type: str or None
            'classification' or 'regression'
            If None, try to guess

        max_sample: int or None
            The maximum number of samples used in the random forest
            If None, unlimited

        subsample: str or np.ndarray or None
            If None, use all samples
        
        bad_treatment: str
            'drop' or 'ignore'
        
        auto_balance: bool
            If True, balance the class by random undersampling
            It only works for classification
        
        check_res: bool
            If True, print the train/test score
        
        return_more: bool
            If True, return more results
            IF False, only return feature_importance and score_test
        '''

        xs, y = self._prepare_ML_data(x_names, y_name, subsample,
                                      bad_treatment)

        if problem_type is None:
            problem_type = guess_problem_type(y)

        if auto_balance:
            if problem_type != 'classification':
                raise ValueError('auto_balance only works for classification')
            xs, y = balance_class(xs, y)

        if not max_sample is None:
            if xs.shape[0] > max_sample:
                print(
                    f'Randomly select {max_sample} samples from {xs.shape[0]} samples'
                )
                subsample = np.random.choice(xs.shape[0],
                                             max_sample,
                                             replace=False)
                xs = xs[subsample]
                y = y[subsample]

        feature_importance, score_test, score_train, rf, X_train, X_test, y_train, y_test = get_RF_importance(
            xs, y, problem_type, return_more=True, **kwargs)

        if check_res:
            print('Check the result:')
            print('  Train score: ', score_train)
            print('  Test score: ', score_test)

        if return_more:
            return feature_importance, score_test, score_train, rf, X_train, X_test, y_train, y_test
        else:
            return feature_importance, score_test

    def get_LDA_projection(self,
                           x_names,
                           y_name,
                           n_components=2,
                           subsample=None,
                           bad_treatment='drop',
                           string_format='.2f',
                           plot=False,
                           return_more=False):

        xs, y = self._prepare_ML_data(x_names, y_name, subsample,
                                      bad_treatment)

        _, lda = get_LDA_projection(xs,
                                    y,
                                    n_components=n_components,
                                    return_more=True)

        def lda_project(X):
            return X @ lda.scalings_

        axis_label_list = []
        for i in range(lda.scalings_.shape[1]):
            axis_label = self.get_linear_combination_string(
                lda.scalings_[:, i], x_names, string_format=string_format)
            axis_label_list.append(axis_label)

        if return_more:
            return axis_label_list, lda_project, lda
        else:
            return axis_label_list, lda_project

    def search_combination_OLS(self,
                               x_names,
                               y_name,
                               n_components=2,
                               allowe_small_n=False,
                               subsample=None,
                               bad_treatment='drop',
                               string_format='.2f',
                               plot=False,
                               metric='mse_resid',
                               is_sigma_clip=False,
                               sigma=3,
                               return_more=False):

        xs, y = self._prepare_ML_data(x_names, y_name, subsample,
                                      bad_treatment)
        best_combination, best_results, results, rank, res_metric = search_combination_OLS(
            xs,
            y,
            n_components=n_components,
            return_more=True,
            is_sigma_clip=is_sigma_clip,
            sigma=sigma,
            metric=metric,
            allowe_small_n=allowe_small_n)

        if plot:
            raise NotImplementedError('plot is not implemented')

        strings = {}
        if return_more:

            for combination in results:
                this_name_list = [''] + [x_names[i] for i in combination]
                strings[combination] = self.get_linear_combination_string(
                    results[combination][0].params,
                    this_name_list,
                    string_format=string_format)
            return strings, best_combination, best_results, results, rank, res_metric
        else:
            this_name_list = [''] + [x_names[i] for i in best_combination]
            best_string = self.get_linear_combination_string(
                best_results[0].params,
                this_name_list,
                string_format=string_format)
            return best_string, best_combination, best_results

    def search_combination_RF(self,
                              x_names,
                              y_name,
                              n_components=2,
                              problem_type=None,
                              allowe_small_n=False,
                              subsample=None,
                              bad_treatment='drop',
                              auto_balance=False,
                              max_sample=None,
                              plot=False,
                              metric=None,
                              return_more=False,
                              CVS_method='grid',
                              param_grid='basic',
                              param_distributions=None,
                              drop_estimator=False,
                              CVS_kwargs=None):

        xs, y = self._prepare_ML_data(x_names, y_name, subsample,
                                      bad_treatment)

        if problem_type is None:
            problem_type = guess_problem_type(y)

        if auto_balance:
            if problem_type != 'classification':
                raise ValueError('auto_balance only works for classification')
            xs, y = balance_class(xs, y)

        if not max_sample is None:
            if xs.shape[0] > max_sample:
                print(
                    f'Randomly select {max_sample} samples from {xs.shape[0]} samples'
                )
                subsample = np.random.choice(xs.shape[0],
                                             max_sample,
                                             replace=False)
                xs = xs[subsample]
                y = y[subsample]

        if problem_type == 'classification':
            if metric is None:
                metric = 'balanced_accuracy'
            best_combination, best_results, results, rank, res_metric = search_combination_RF_cls(
                xs,
                y,
                n_components=n_components,
                allowe_small_n=allowe_small_n,
                return_more=True,
                metric=metric,
                CVS_method=CVS_method,
                param_grid=param_grid,
                param_distributions=param_distributions,
                drop_estimator=drop_estimator,
                CVS_kwargs=CVS_kwargs)
        elif problem_type == 'regression':
            if metric is None:
                metric = 'mse_resid'
            best_combination, best_results, results, rank, res_metric = search_combination_RF_reg(
                xs,
                y,
                n_components=n_components,
                allowe_small_n=allowe_small_n,
                return_more=True,
                metric=metric,
                CVS_method=CVS_method,
                param_grid=param_grid,
                param_distributions=param_distributions,
                drop_estimator=drop_estimator,
                CVS_kwargs=CVS_kwargs)

        if plot:
            raise NotImplementedError('plot is not implemented')

        strings = {}
        if return_more:
            for combination in results:
                this_name_list = [x_names[i] for i in combination]
                strings[combination] = self.get_func_combination_string(
                    this_name_list, 'RF', with_unit=False)
            return strings, best_combination, best_results, results, rank, res_metric
        else:
            this_name_list = [x_names[i] for i in best_combination]
            best_string = self.get_func_combination_string(this_name_list,
                                                           'RF',
                                                           with_unit=False)
            return best_string, best_combination, best_results

    def _prepare_ML_data(self, x_names, y_name, subsample, bad_treatment):
        x_names = string_to_list(x_names)
        if y_name in x_names:
            warnings.warn(f'y_name: {y_name} is in x_names: {x_names}')
        xs = self.get_data_by_names(x_names)
        y = self.get_data_by_name(y_name)
        _subsample = self.get_subsample(subsample)
        xs = xs[_subsample]
        y = y[_subsample]

        if is_bool(y):
            y = y.astype(int)

        if bad_treatment == 'drop':
            xs, y = remove_bad([xs, y])
        else:
            raise NotImplementedError(
                'bad_treatment other than drop is not implemented')

        return xs, y


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


def guess_problem_type(y):
    print('problem_type is not specified, try to guess:')
    print('  If y is float, problem_type is regression')
    print('  If y is int or bool, problem_type is classification')
    if is_float(y):
        return 'regression'
    elif is_int(y):
        return 'classification'
    else:
        raise ValueError(
            'Can not guess problem_type, please specify problem_type')
