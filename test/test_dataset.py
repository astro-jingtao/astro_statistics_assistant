import pytest
import numpy as np
from asa.dataset import Dataset
from asa.dataset import parse_inequality


class TestGetitem:
    def test_getitem(self):
        x = np.arange(10)
        y = np.arange(10) * 2
        z = np.arange(10) * 3
        dataset = Dataset(
            np.array([x, y, z]).T, ['x', 'y', 'z'],
            ['x label', 'y label', 'z label'])

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
        assert 'key[0] can not be string or list of string' == str(
            excinfo.value)

    def test_parse_inequality(self):

        assert (parse_inequality("1<=2<=x") == ['1', '<=', '2', '<=', 'x'])
        assert (parse_inequality("1>2>=x") == ['1', '>', '2', '>=', 'x'])
        assert (parse_inequality("1 <= 2 <= x") == ['1', '<=', '2', '<=', 'x'])
        assert parse_inequality("10<20>30>=40<=50") == [
            '10', '<', '20', '>', '30', '>=', '40', '<=', '50'
        ]
        assert parse_inequality("5<=10") == ['5', '<=', '10']
        assert parse_inequality("15>=5") == ['15', '>=', '5']
        assert parse_inequality("100<200") == ['100', '<', '200']
        assert parse_inequality("30>20") == ['30', '>', '20']
        assert parse_inequality("x<100") == ['x', '<', '100']
        assert parse_inequality("a>=b") == ['a', '>=', 'b']

    def test_inequality_to_subsample(self):

        x = np.arange(10)
        y = np.arange(10) * 2
        z = np.arange(10) * 3
        dataset = Dataset(
            np.array([x, y, z]).T, ['x', 'y', 'z'],
            ['x label', 'y label', 'z label'])

        debug = False

        assert np.array_equal(
            dataset.inequality_to_subsample("x<5", debug=debug),
            np.array(x < 5))
        assert np.array_equal(
            dataset.inequality_to_subsample("x<=5", debug=debug),
            np.array(x <= 5))
        assert np.array_equal(
            dataset.inequality_to_subsample("x>y", debug=debug),
            np.array(x > y))

    def test_check_same_length(self):

        x = np.arange(10)
        y = np.arange(10) * 2
        z = np.arange(10) * 3

        with pytest.raises(ValueError) as excinfo:
            Dataset(
                np.array([x, y, z]).T, ['x', 'y'],
                ['x label', 'y label', 'z label'])
        assert 'data and names have different length' == str(excinfo.value)

        with pytest.raises(ValueError) as excinfo:
            Dataset(
                np.array([x, y, z]).T, ['x', 'y', 'z'], ['x label', 'y label'])
        assert 'data and labels have different length' == str(excinfo.value)

        Dataset(
                np.array([x, y, z]).T, ['x', 'y', 'z'],
                ['x label', 'y label', 'z label'])
