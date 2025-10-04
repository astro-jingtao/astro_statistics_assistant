import numpy as np
import pytest
from statsmodels.stats.weightstats import DescrStatsW

from asa.weighted_statistic import *


class TestWeightedStatistic:

    def test_effect_sample_size(self):
        w = np.ones(100)
        assert get_effect_sample_size(w) == 100

    def test_shape(self):
        x = np.random.normal(size=100)
        w = np.random.uniform(size=100)

        assert median(x, w).shape == ()
        assert median(x).shape == ()

        assert mean(x, w).shape == ()
        assert mean(x).shape == ()

        assert std(x, w).shape == ()
        assert std(x).shape == ()

        assert std_mean(x, w).shape == ()
        assert std_mean(x).shape == ()

        assert std_median(x, w).shape == ()
        assert std_median(x).shape == ()

        assert quantile(x, w).shape == ()
        assert quantile(x).shape == ()

    def test_res(self):
        x = np.random.normal(size=100)
        w = np.random.uniform(size=100)

        assert mean(x) == np.mean(x)
        assert median(x) == np.median(x)
        assert std(x) == np.std(x)
        assert std_mean(x) == np.std(x) / np.sqrt(len(x))

        ds = DescrStatsW(x, weights=w)
        assert np.isclose(mean(x, w), ds.mean)

        assert np.isclose(std(x, ddof=1), np.std(x, ddof=1))
        assert np.isclose(std(x), std(x, w=np.ones_like(x)))
        assert np.isclose(std(x, ddof=1), std(x, w=np.ones_like(x), ddof=1))
        assert np.isclose(std(x, ddof=1), std(x, w=np.ones_like(x) * 2,
                                              ddof=1))

        assert np.isclose(std_mean(x), std_mean(x, w=np.ones_like(x)))
        assert np.isclose(std_mean(x, ddof=1),
                          std_mean(x, w=np.ones_like(x), ddof=1))
        assert np.isclose(std_mean(x, ddof=1),
                          std_mean(x, w=np.ones_like(x) * 2, ddof=1))

        assert np.isclose(std_median(x), std_median(x, w=np.ones_like(x)))
        assert np.isclose(std_median(x), std_median(x, w=np.ones_like(x) * 2))


class TestAxis:

    def test_get_effect_sample_size(self):

        w = np.random.uniform(size=(10, 100))
        _n_axis = get_effect_sample_size(w, axis=1)

        assert _n_axis.shape == (10, )
        for i in range(10):
            assert np.isclose(get_effect_sample_size(w[i]), _n_axis[i])

    def test_mean(self):

        x = np.random.normal(size=(10, 100))
        w = np.random.uniform(size=(10, 100))

        _m_axis = mean(x, w=w, axis=1)

        for i in range(10):
            assert np.isclose(mean(x[i], w=w[i]), _m_axis[i])

        # broadcasting multi x single w
        _m_axis = mean(x, w=w[0], axis=1)
        for i in range(10):
            assert np.isclose(mean(x[i], w=w[0]), _m_axis[i])

        # broadcasting single x multi w
        _m_axis = mean(x[0], w=w, axis=1)
        for i in range(10):
            assert np.isclose(mean(x[0], w=w[i]), _m_axis[i])

    def test_std(self):

        x = np.random.normal(size=(10, 100))
        w = np.random.uniform(size=(10, 100))

        _s_axis = std(x, w=w, axis=1)

        for i in range(10):
            assert np.isclose(std(x[i], w=w[i]), _s_axis[i])

        # broadcasting multi x single w
        _s_axis = std(x, w=w[0], axis=1)
        for i in range(10):
            assert np.isclose(std(x[i], w=w[0]), _s_axis[i])

        # broadcasting single x multi w
        _s_axis = std(x[0], w=w, axis=1)
        for i in range(10):
            assert np.isclose(std(x[0], w=w[i]), _s_axis[i])

        # ddof > N
        _s_axis_all_nan = std(x, w=w, axis=1, ddof=100)
        print(_s_axis_all_nan)
        assert _s_axis_all_nan.shape == (10, )
        for i in range(10):
            assert np.isnan(_s_axis_all_nan[i])


class TestBroadcastArrays:

    def test_single_array(self):
        arr = np.array([1, 2, 3])
        result = broadcast_arrays(arr)
        assert len(result) == 1
        np.testing.assert_array_equal(result[0], arr)

    def test_two_broadcastable_arrays(self):
        arr1 = np.array([[1, 2, 3], [4, 5, 6]])
        arr2 = np.array([[10], [20]])
        result = broadcast_arrays(arr1, arr2)
        expected_shape = (2, 3)
        assert len(result) == 2
        assert result[0].shape == expected_shape
        assert result[1].shape == expected_shape
        np.testing.assert_array_equal(result[0], [[1, 2, 3], [4, 5, 6]])
        np.testing.assert_array_equal(result[1], [[10, 10, 10], [20, 20, 20]])

    def test_three_broadcastable_arrays(self):
        arr1 = np.array([[1, 2, 3], [4, 5, 6]])
        arr2 = np.array([[10], [20]])
        arr3 = np.array([[1], [2]])
        result = broadcast_arrays(arr1, arr2, arr3)
        expected_shape = (2, 3)
        assert len(result) == 3
        assert result[0].shape == expected_shape
        assert result[1].shape == expected_shape
        assert result[2].shape == expected_shape
        np.testing.assert_array_equal(result[0], [[1, 2, 3], [4, 5, 6]])
        np.testing.assert_array_equal(result[1], [[10, 10, 10], [20, 20, 20]])
        np.testing.assert_array_equal(result[2], [[1, 1, 1], [2, 2, 2]])

    def test_zero2two_broadcaste(self):
        arr1 = np.array([[1, 2], [3, 4]])
        arr2 = np.array([[1]])
        result = broadcast_arrays(arr1, arr2)
        expected_shape = (2, 2)
        assert len(result) == 2
        assert result[0].shape == expected_shape
        assert result[1].shape == expected_shape
        np.testing.assert_array_equal(result[0], [[1, 2], [3, 4]])
        np.testing.assert_array_equal(result[1], [[1, 1], [1, 1]])

    def test_empty_input(self):
        result = broadcast_arrays()
        assert result == []

    def test_different_shapes(self):
        arr1 = np.array([[1, 2, 3]])
        arr2 = np.array([[10], [20]])
        arr3 = np.array([[1]])
        result = broadcast_arrays(arr1, arr2, arr3)
        expected_shape = (2, 3)
        assert len(result) == 3
        assert result[0].shape == expected_shape
        assert result[1].shape == expected_shape
        assert result[2].shape == expected_shape
        np.testing.assert_array_equal(result[0], [[1, 2, 3], [1, 2, 3]])
        np.testing.assert_array_equal(result[1], [[10, 10, 10], [20, 20, 20]])
        np.testing.assert_array_equal(result[2], [[1, 1, 1], [1, 1, 1]])
