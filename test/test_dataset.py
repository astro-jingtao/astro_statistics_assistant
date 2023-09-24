import pytest
import numpy as np
from asa.dataset import Dataset

class TestGetitem:

    def test_getitem(self):
        x = np.arange(10)
        y = np.arange(10) * 2
        z = np.arange(10) * 3
        dataset = Dataset(np.array([x, y, z]).T, ['x', 'y', 'z'], ['x label', 'y label', 'z label'])

        assert np.array_equal(dataset[0], np.array([x[0], y[0], z[0]]))
        assert np.array_equal(dataset[:, :], np.array([x, y, z]).T)
        assert np.array_equal(dataset[:, 0], x)
        assert np.array_equal(dataset[:, ['x', 'y']], np.array([x, y]).T)
        assert np.array_equal(dataset[:, 'x'], x)
        assert np.array_equal(dataset[['x', 'y']], np.array([x, y]).T)

        with pytest.raises(ValueError) as excinfo:
            dataset[0, 0, 0]
        assert 'key should be a tuple of length 2' == str(excinfo.value)

        with pytest.raises(ValueError) as excinfo:
            dataset['x', 0]
        assert 'key[0] can not be string or list of string' == str(excinfo.value)


