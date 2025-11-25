import sys

import numpy as np
import pytest

from asa.utils import (balance_class, is_bool, is_float, is_int, list_reshape,
                       remove_bad, to_little_endian, auto_set_range,
                       all_subsample, deduplicate)


class TestUtils:

    def test_list_reshape(self):
        lst = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        shape = (3, 3)
        expected_output = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        assert list_reshape(lst, shape) == expected_output

        lst = [1, 2, 3, 4, 5, 6, 7, 8]
        shape = (2, 4)
        expected_output = [[1, 2, 3, 4], [5, 6, 7, 8]]
        assert list_reshape(lst, shape) == expected_output

    def test_is_float(self):

        assert is_float(0.1)
        assert is_float(np.float32(0.1))
        assert is_float(np.float64(0.1))
        assert is_float(np.array([0.1, 0.2, 0.3], dtype=np.float32))
        assert is_float(np.array([0.1, 0.2, 0.3], dtype=np.float64))
        assert is_float([0.1, 0.2, 0.3])

        assert not is_float(1)
        assert not is_float(np.int32(1))
        assert not is_float(np.int64(1))
        assert not is_float(np.array([1, 2, 3], dtype=np.int32))
        assert not is_float(np.array([1, 2, 3], dtype=np.int64))

        assert not is_float('a')
        assert not is_float(np.array(['a', 'b', 'c']))

        assert not is_float([1, 2, 3])
        assert not is_float([0.1, 0.2, 'a'])

        assert not is_float(np.ones(10, dtype=bool))
        assert not is_float(np.zeros(10, dtype=str))

    def test_is_int(self):

        assert not is_int(0.1)
        assert not is_int(np.float32(0.1))
        assert not is_int(np.float64(0.1))
        assert not is_int(np.array([0.1, 0.2, 0.3], dtype=np.float32))
        assert not is_int(np.array([0.1, 0.2, 0.3], dtype=np.float64))
        assert not is_int([0.1, 0.2, 0.3])

        assert is_int(1)
        assert is_int(np.int32(1))
        assert is_int(np.int64(1))
        assert is_int(np.array([1, 2, 3], dtype=np.int32))
        assert is_int(np.array([1, 2, 3], dtype=np.int64))
        assert is_int([1, 2, 3])

        assert not is_int('a')
        assert not is_int(np.array(['a', 'b', 'c']))
        assert not is_int([0.1, 0.2, 'a'])
        assert not is_int(np.ones(10, dtype=bool))
        assert not is_int(np.zeros(10, dtype=str))

    def test_is_bool(self):

        assert not is_bool(0.1)
        assert not is_bool(np.float32(0.1))
        assert not is_bool(np.float64(0.1))
        assert not is_bool(np.array([0.1, 0.2, 0.3], dtype=np.float32))
        assert not is_bool(np.array([0.1, 0.2, 0.3], dtype=np.float64))
        assert not is_bool([0.1, 0.2, 0.3])

        assert not is_bool(1)
        assert not is_bool(np.int32(1))
        assert not is_bool(np.int64(1))
        assert not is_bool(np.array([1, 2, 3], dtype=np.int32))
        assert not is_bool(np.array([1, 2, 3], dtype=np.int64))
        assert not is_bool([1, 2, 3])

        assert not is_bool('a')
        assert not is_bool(np.array(['a', 'b', 'c']))
        assert not is_bool([0.1, 0.2, 'a'])

        assert is_bool(True)
        assert is_bool(np.ones(10, dtype=bool))
        assert is_bool(np.zeros(10, dtype=bool))
        assert is_bool([True, False, True])

    def test_balance_class(self):

        x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        y = np.array([0, 1, 2])

        x_balanced, y_balanced = balance_class(x, y, random_state=0)

        assert x_balanced.shape == (3, 3)
        assert y_balanced.shape == (3, )

        x = np.random.normal(size=(100, 10))
        y = np.random.randint(0, 4, size=(100, ))
        y[y == 3] = 2

        x_balanced, y_balanced = balance_class(x, y, random_state=0)

        _, counts = np.unique(y, return_counts=True)

        assert x_balanced.shape == (3 * counts.min(), 10)
        assert y_balanced.shape == (3 * counts.min(), )

        for i in range(3):
            assert (y_balanced == i).sum() == (y_balanced == 0).sum()

    def test_auto_range(self):
        assert np.array_equal(auto_set_range([1, 2, 3], [4, 5, 6], None, None),
                              [[1, 3], [4, 6]])

        assert np.array_equal(auto_set_range([1], [4], None, None),
                              [[0.5, 1.5], [3.5, 4.5]])

        assert np.array_equal(auto_set_range([], [], None, None),
                              [[0, 1], [0, 1]])


class TestToLittle:

    expected_byteorder = '=' if sys.byteorder == 'little' else '<'

    def test_big_endian_input(self):  # sourcery skip: class-extract-method
        # Test input with big-endian byte order
        arr = np.array([1, 2, 3], dtype='>f')
        result = to_little_endian(arr)
        # Verify that the byte order is converted to little-endian
        assert result.dtype.byteorder == '<'

    def test_little_endian_input(self):
        # Test input that is already in little-endian format
        arr = np.array([1, 2, 3], dtype='<f')
        result = to_little_endian(arr)
        # Check if the byte order remains unchanged
        assert result.dtype.byteorder == self.expected_byteorder

    def test_native_endian_input(self):
        # Test input using the system's default byte order when the system is little-endian
        arr = np.array([1, 2, 3], dtype='=f')
        result = to_little_endian(arr)
        # Check if the byte order is set correctly based on system's endianess
        assert result.dtype.byteorder == self.expected_byteorder

    def test_single_byte_dtype(self):
        # Test with a single-byte data type where byte order is irrelevant
        arr = np.array([1, 2, 3], dtype='uint8')
        result = to_little_endian(arr)
        # Ensure that the byte order remains unchanged ('|' indicates not applicable)
        assert result.dtype.byteorder == '|'


class TestAllSubsample:

    def test_empty_list(self):
        # Test with an empty list
        xs = []
        idx = 0
        assert all_subsample(xs, idx) == []

    def test_none_element(self):
        # Test with a list containing None
        xs = [None]
        idx = 0
        assert all_subsample(xs, idx) == [None]

    def test_1d_array(self):
        # Test with a list containing a 1D numpy array
        xs = [np.array([1, 2, 3])]
        idx = [0, 2]
        assert np.array_equal(all_subsample(xs, idx)[0], np.array([1, 3]))

    def test_2d_array(self):
        # Test with a list containing a 2D numpy array
        xs = [np.array([[1, 2], [3, 4], [5, 6]])]
        idx = [0, 1]
        expected = [np.array([[1, 2], [3, 4]])]
        result = all_subsample(xs, idx)

        assert len(result) == 1
        assert np.allclose(result[0], expected[0])


class TestRemoveBad:

    def test_nan(self):
        x = np.random.normal(size=100)
        x[10:14] = np.nan
        assert np.array_equal(remove_bad([x])[0], x[~np.isnan(x)])

    def test_inf(self):
        x = np.random.normal(size=100)
        x[10:14] = np.inf
        assert np.array_equal(remove_bad([x])[0], x[~np.isinf(x)])

    def test_mixed(self):
        x = np.random.normal(size=100)
        x[10:14] = np.nan
        y = np.random.normal(size=100)
        y[80:84] = np.inf
        _x, _y = remove_bad([x, y])
        assert np.array_equal(_x, x[~np.isnan(x) & ~np.isinf(y)])
        assert np.array_equal(_y, y[~np.isnan(x) & ~np.isinf(y)])

    def test_2d_array(self):
        X = np.random.normal(size=(100, 10))
        X[10:14, 0] = np.nan
        assert np.array_equal(remove_bad([X])[0], X[~np.isnan(X[:, 0])])

    def test_none_element(self):
        x = np.random.normal(size=100)
        y = np.random.normal(size=100)
        y[80:84] = np.nan
        z = None
        _x, _y, _z = remove_bad([x, y, z])
        assert np.array_equal(_x, x[~np.isnan(y)])
        assert np.array_equal(_y, y[~np.isnan(y)])
        assert _z is None

    def test_transpose(self):
        x = np.random.normal(size=100)
        X = np.random.normal(size=(100, 10))
        X[10:14, 0] = np.nan
        X = X.T
        _x, _X = remove_bad([x, X], to_transpose=[1])
        is_good = ~np.isnan(X[0, :])
        assert np.array_equal(_x, x[is_good])
        assert np.array_equal(_X, X[:, is_good])


class TestDeduplicate:

    def test_small_difference(self):
        x_dedup = deduplicate([1., 1., 1., 1.05], max_dx=0.1)
        assert np.all(np.diff(x_dedup) > 0)

    def test_all_same(self):
        x_dedup = deduplicate([1., 1., 1., 1.], max_dx=0.1)
        assert np.all(np.diff(x_dedup) > 0)

    def test_not_sorted(self):
        with pytest.raises(ValueError) as excinfo:
            deduplicate([1., 2., 1., 3.])
            assert 'Input array must be sorted' in str(excinfo.value)
            