import sys

import numpy as np
import pytest

from asa.utils import (balance_class, is_bool, is_float, is_int, list_reshape,
                       remove_bad, to_little_endian)


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

    def test_remove_bad(self):
        x = np.random.normal(size=100)
        x[10:14] = np.nan
        assert np.array_equal(remove_bad([x])[0], x[~np.isnan(x)])

        x = np.random.normal(size=100)
        x[10:14] = np.inf
        assert np.array_equal(remove_bad([x])[0], x[~np.isinf(x)])

        x = np.random.normal(size=100)
        x[10:14] = np.nan
        y = np.random.normal(size=100)
        y[80:84] = np.inf
        _x, _y = remove_bad([x, y])
        assert np.array_equal(_x, x[~np.isnan(x) & ~np.isinf(y)])
        assert np.array_equal(_y, y[~np.isnan(x) & ~np.isinf(y)])

        X = np.random.normal(size=(100, 10))
        X[10:14, 0] = np.nan
        assert np.array_equal(remove_bad([X])[0], X[~np.isnan(X[:, 0])])

        x = np.random.normal(size=100)
        y = np.random.normal(size=100)
        y[80:84] = np.nan
        z = None
        _x, _y, _z = remove_bad([x, y, z])
        assert np.array_equal(_x, x[~np.isnan(y)])
        assert np.array_equal(_y, y[~np.isnan(y)])
        assert _z is None


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
