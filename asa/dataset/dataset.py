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

from .plot_dataset import PlotDataset

# TODO: DF to AASTeX tabel. Maybe ref to: https://github.com/liuguanfu1120/Excel-to-AASTeX/blob/main/xlsx-to-AAS-table.ipynb


class Dataset(PlotDataset):

    # -- Note -- that all values passed to plot_xxx should be numpy array, not series

    # TODO: inherit the doc string of wrapped methods

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
