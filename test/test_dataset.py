import pytest
import numpy as np
import astropy.units as u
from asa.dataset import Dataset, Labels
from asa.dataset.inequality_utlis import parse_inequality, parse_and_or, parse_op
import asa.uncertainty as unc


def gen_dataset():
    x = np.arange(10)
    y = np.arange(10) * 2
    z = np.arange(10) * 3
    dataset = Dataset(
        np.array([x, y, z]).T, ['x', 'y', 'z'],
        ['x label', 'y label', 'z label'])
    return dataset, x, y, z


class TestDataset:

    def test_getitem(self):
        dataset, x, y, z = gen_dataset()

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

    def test_check_same_length(self):

        _, x, y, z = gen_dataset()

        # this error is handled by pd.DataFrame
        with pytest.raises(ValueError) as excinfo:
            Dataset(
                np.array([x, y, z]).T, ['x', 'y'],
                ['x label', 'y label', 'z label'])
        assert 'Shape of passed values is (10, 3), indices imply (10, 2)' == str(
            excinfo.value)

        with pytest.raises(IndexError) as excinfo:
            Dataset(
                np.array([x, y, z]).T, ['x', 'y', 'z'], ['x label', 'y label'])
        assert 'list index out of range' == str(excinfo.value)

        Dataset(
            np.array([x, y, z]).T, ['x', 'y', 'z'],
            ['x label', 'y label', 'z label'])

    def test_setitem(self):

        dataset, x, y, z = gen_dataset()
        dataset['x'] = x * 2
        assert np.array_equal(dataset['x'], x * 2)
        dataset[['x', 'y']] = np.array([x * 2, y * 3]).T
        assert np.array_equal(dataset['x'], x * 2)
        assert np.array_equal(dataset['y'], y * 3)

        dataset, x, y, z = gen_dataset()
        dataset[:, 'x'] = x * 2
        assert np.array_equal(dataset['x'], x * 2)

        dataset, x, y, z = gen_dataset()
        dataset[:, ['x', 'y']] = np.array([x * 2, y * 3]).T
        assert np.array_equal(dataset['x'], x * 2)
        assert np.array_equal(dataset['y'], y * 3)

        dataset, x, y, z = gen_dataset()
        dataset['x2'] = x * 2
        assert np.array_equal(dataset['x2'], x * 2)
        # two new columns
        dataset[['x3', 'x4']] = np.array([x * 2, y * 3]).T
        assert np.array_equal(dataset['x3'], x * 2)
        assert np.array_equal(dataset['x4'], y * 3)
        # new column and update
        dataset[['x3', 'x5']] = np.array([x, y]).T
        assert np.array_equal(dataset['x3'], x)
        assert np.array_equal(dataset['x5'], y)

    def test_construct(self):
        _, x, y, z = gen_dataset()

        # label as dict
        dataset = Dataset(np.array([x, y, z]).T,
                          names=['x', 'y', 'z'],
                          labels={
                              'x': 'x_label',
                              'y': 'y_label',
                              'z': 'z_label'
                          })
        assert dataset.labels == {
            'x': 'x_label',
            'y': 'y_label',
            'z': 'z_label'
        }

        dataset = Dataset(np.array([x, y, z]).T,
                          names=['x', 'y', 'z'],
                          labels={
                              'x': 'x_label',
                              'y': 'y_label'
                          })
        assert dataset.labels == {'x': 'x_label', 'y': 'y_label'}

    def test_update(self):

        dataset, x, y, z = gen_dataset()
        assert dataset.labels == {
            'x': 'x label',
            'y': 'y label',
            'z': 'z label'
        }

        dataset.update_labels({'x': 'xxx', 'y': 'yyy'})
        assert dataset.labels == {'x': 'xxx', 'y': 'yyy', 'z': 'z label'}

        dataset.update_names({'x': 'x1'})
        assert np.array_equal(dataset.names, np.array(['x1', 'y', 'z']))
        assert np.array_equal(np.asarray(dataset.data.columns), dataset.names)
        assert np.array_equal(dataset['x1'], x)

        dataset.update_names({'x1': '777', 'y': 'I_love_u', 'z': 'good'})
        assert np.array_equal(dataset.names,
                              np.array(['777', 'I_love_u', 'good']))
        assert np.array_equal(np.asarray(dataset.data.columns), dataset.names)
        assert np.array_equal(dataset['777'], x)
        assert np.array_equal(dataset['I_love_u'], y)
        assert np.array_equal(dataset['good'], z)

    @pytest.mark.filterwarnings("ignore::RuntimeWarning")  # np.log10(0)
    def test_get_range_by_name(self):

        _, x, y, z = gen_dataset()
        dataset = Dataset(np.array([x, y, z]).T, ['x', 'y', 'z'],
                          ['x label', 'y label', 'z label'],
                          ranges={'x': [0, 9]})

        assert dataset.get_range_by_name('x') == [0, 9]
        assert dataset.get_range_by_name('log10@x') is None
        assert dataset.get_range_by_name('y') is None

        dataset = Dataset(np.array([x, y, z]).T, ['x', 'y', 'z'],
                          ['x label', 'y label', 'z label'],
                          ranges={'x': [1, 9]})

        assert dataset.get_range_by_name('log10@x') == [
            np.log10(1), np.log10(9)
        ]

    def test_get_data_by_name(self):

        dataset, x, y, z = gen_dataset()

        assert np.array_equal(dataset.get_data_by_name('x'), x)
        assert np.array_equal(dataset.get_data_by_names(['x', 'y']),
                              np.array([x, y]).T)

        x = np.arange(10) + 1
        x_err = (np.arange(10) + 1) * 0.1
        dataset = Dataset(np.array([x, x_err]).T, ['x', 'x_err'])
        assert np.array_equal(dataset.get_data_by_name('x'), x)

        # --- with unit
        assert np.array_equal(dataset.get_data_by_name('x', with_unit=True), x)
        dataset.update_units({'x': u.cm})
        assert np.array_equal(dataset.get_data_by_name('x', with_unit=True),
                              x * u.cm)
        # do not with unit for err, snr, or op
        assert np.array_equal(
            dataset.get_data_by_name('x_err', with_unit=True), x_err)
        assert np.array_equal(
            dataset.get_data_by_name('x_snr', with_unit=True),
            np.abs(x / x_err))
        assert np.array_equal(
            dataset.get_data_by_name('log10@x', with_unit=True), np.log10(x))

        assert np.array_equal(dataset.get_data_by_name('x_err'), x_err)
        assert np.allclose(dataset.get_data_by_name('x_snr'),
                           np.abs(x / x_err))

        assert np.allclose(dataset.get_data_by_name('log10@x'), np.log10(x))

        assert np.allclose(dataset.get_data_by_name('x_snr'),
                           np.abs(x / x_err))
        assert np.allclose(dataset.get_data_by_name('log10@x_err'),
                           unc.log10(x, x_err))
        assert np.allclose(dataset.get_data_by_name('log10@x_snr'),
                           np.abs(np.log10(x) / unc.log10(x, x_err)))

        dataset = Dataset(np.array([x, x_err]).T, ['x', 'x_errerr'],
                          err_postfix='errerr')
        assert np.array_equal(dataset.get_data_by_name('x_errerr'), x_err)

        with pytest.raises(ValueError) as excinfo:
            dataset.get_data_by_name('x_err')
        assert "'x_err' is not in list" == str(excinfo.value)

        x_snr = x / x_err
        dataset = Dataset(np.array([x, x_snr]).T, ['x', 'x_snr'])
        assert np.array_equal(dataset.get_data_by_name('x_snr'), x_snr)
        assert np.allclose(dataset.get_data_by_name('x_err'), x_err)
        assert np.allclose(dataset.get_data_by_name('log10@x_snr'),
                           unc.log10_snr(x, x_snr))
        assert np.allclose(dataset.get_data_by_name('log10@x_err'),
                           unc.log10(x, x_err))

        # potential false positive
        dataset = Dataset(
            np.array([x, x_snr, x, x]).T, ['x', 'x_snr', 'snr_x', 'xsnr'])
        assert np.array_equal(dataset.get_data_by_name('snr_x'), x)
        assert np.array_equal(dataset.get_data_by_name('xsnr'), x)

    def test_remove_postfix(self):
        dataset, x, y, z = gen_dataset()
        assert dataset.remove_snr_postfix('x_snr') == 'x'
        with pytest.raises(ValueError) as excinfo:
            dataset.remove_snr_postfix('x')
        assert "x does not end with _snr" == str(excinfo.value)

        assert dataset.remove_err_postfix('x_err') == 'x'
        with pytest.raises(ValueError) as excinfo:
            dataset.remove_err_postfix('x')
        assert "x does not end with _err" == str(excinfo.value)

    def test_is_legal_name(self):
        dataset, x, y, z = gen_dataset()
        assert dataset.is_legal_name('x')
        assert dataset.is_legal_name('y')
        assert dataset.is_legal_name('z')
        assert dataset.is_legal_name('x_err')
        assert dataset.is_legal_name('x_snr')
        assert dataset.is_legal_name('log10@x')
        assert dataset.is_legal_name('log10@x_err')

        assert not dataset.is_legal_name('t')
        assert not dataset.is_legal_name('t_err')
        assert not dataset.is_legal_name('t_snr')
        assert not dataset.is_legal_name('log10@t')
        assert not dataset.is_legal_name('log10@t_err')

    def test_random_subsample(self):
        dataset, x, y, z = gen_dataset()
        assert len(dataset.random_subsample(5)) == 5
        assert dataset.random_subsample(5, as_bool=True).sum() == 5

        assert len(dataset.random_subsample(0.5)) == 5

        for _ in range(10):
            subsample = dataset.random_subsample(3, input_subsample='x<5')
            assert len(subsample) == 3
            assert np.all(x[subsample] < 5)

    def test_get_label_by_name(self):
        dataset, x, y, z = gen_dataset()
        dataset.unit_labels = {'x': 'a'}
        assert dataset.get_label_by_name('x') == 'x label a'
        assert dataset.get_label_by_name('y') == 'y label'
        assert dataset.get_label_by_name('x', with_unit=False) == 'x label'
        assert dataset.get_label_by_name('log10@x') == r'$\log$x label a'

        assert dataset.get_label_by_name(
            'log10@x', op_bracket='[{label}]') == r'$\log$[x label] a'
        assert dataset.get_label_by_name(
            'log10@x', op_bracket='({label})') == r'$\log$(x label) a'

        dataset.update_labels({"log10@x": 'log10x'})
        assert dataset.get_label_by_name('log10@x',
                                         with_unit=False) == 'log10x'

    def test_get_unit_by_name(self):
        dataset, x, y, z = gen_dataset()
        assert dataset.get_unit_by_name('x') == 1
        assert dataset.get_unit_by_name('y') == 1

        dataset.update_units({'x': u.cm})
        assert dataset.get_unit_by_name('x') == u.cm

    def test_add_col(self):
        dataset, x, y, z = gen_dataset()
        dataset.add_col(x * 2, 'x2')
        assert np.array_equal(dataset['x2'], x * 2)

        # should raise error
        with pytest.raises(ValueError) as excinfo:
            dataset.add_col(x * u.cm, 'x3')

        dataset.update_units({'x3': u.cm, 'x4': u.cm})
        dataset.add_col(x * u.cm, 'x3')
        assert np.array_equal(dataset['x3'], x)
        dataset.add_col(x * u.m, 'x4')
        assert np.array_equal(dataset['x4'], x * 100)

        dataset.add_col([x**2, 2**x], ['x6', 'x7'])
        assert np.array_equal(dataset['x6'], x**2)
        assert np.array_equal(dataset['x7'], 2**x)

    def test_short_name(self):
        dataset, x, y, z = gen_dataset()
        assert np.array_equal(dataset.gdn('x'), dataset.get_data_by_name('x'))
        assert np.array_equal(dataset.gdns(['x', 'y']),
                              dataset.get_data_by_names(['x', 'y']))
        assert dataset.gln('x') == dataset.get_label_by_name('x')
        assert dataset.glns(['x',
                             'y']) == dataset.get_labels_by_names(['x', 'y'])


class TestDatasetInequality:

    def test_inequality_to_subsample(self):

        dataset, x, y, z = gen_dataset()

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
        assert np.array_equal(
            dataset.inequality_to_subsample("x==y", debug=debug),
            np.array(x == y))
        assert np.array_equal(
            dataset.inequality_to_subsample("x==5", debug=debug),
            np.array(x == 5))

        dataset['t'] = x + 2
        assert np.array_equal(
            dataset.inequality_to_subsample("log10@t<np.log10(5)",
                                            debug=debug),
            np.array(np.log10(x + 2) < np.log10(5)))
        assert np.array_equal(
            dataset.inequality_to_subsample("square@t<3", debug=debug),
            np.array(np.square(x + 2) < 3))

        # operation
        assert np.array_equal(
            dataset.inequality_to_subsample("x<=3+2", debug=debug),
            np.array(x <= 5))
        assert np.array_equal(
            dataset.inequality_to_subsample("x<=(z+y)/2", debug=debug),
            np.array(x <= (z + y) / 2))
        assert np.array_equal(
            dataset.inequality_to_subsample("x<=(1+2)*2", debug=debug),
            np.array(x <= 6))
        assert np.array_equal(
            dataset.inequality_to_subsample("x<=1+2*2", debug=debug),
            np.array(x <= 5))
        assert np.array_equal(
            dataset.inequality_to_subsample("x/5<=1", debug=debug),
            np.array(x <= 5))
        assert np.array_equal(
            dataset.inequality_to_subsample("(x + 1)/5<=1", debug=debug),
            np.array(x <= 4))
        assert np.array_equal(
            dataset.inequality_to_subsample("((x + 1) + 2)/5<=1", debug=debug),
            np.array(x <= 2))
        assert not np.array_equal(
            dataset.inequality_to_subsample("x + 1/5<=1", debug=debug),
            np.array(x <= 4))
        assert np.array_equal(
            dataset.inequality_to_subsample("x + y<=5", debug=debug),
            np.array((x + y) <= 5))

        # and, or
        assert np.array_equal(
            dataset.inequality_to_subsample("x > 3 & y > 5", debug=debug),
            np.array((x > 3) & (y > 5)))
        assert np.array_equal(
            dataset.inequality_to_subsample("x > 3 | y > 5", debug=debug),
            np.array((x > 3) | (y > 5)))
        assert np.array_equal(
            dataset.inequality_to_subsample("x > 3 | y == 1", debug=debug),
            np.array((x > 3) | (y == 1)))
        assert np.array_equal(
            dataset.inequality_to_subsample("[x > 3 & [ y > 5 | z < 2]]",
                                            debug=debug),
            np.array((x > 3) & ((y > 5) | (z < 2))))

        # complex
        assert np.array_equal(
            dataset.inequality_to_subsample(
                "[(x + 1)/2 > 3 & [ y > 5/2 | z < 2+1]]", debug=debug),
            np.array((x > 5) & ((y > 2.5) | (z < 3))))

        # bool
        dataset['x_bool'] = x > 5
        dataset['y_bool'] = y > 5
        assert np.array_equal(
            dataset.inequality_to_subsample("x_bool & y_bool", debug=debug),
            np.array((x > 5) & (y > 5)))

        # bool and binary
        dataset['x_bin'] = dataset['x_bool'].astype(int)
        dataset['y_bin'] = dataset['y_bool'].astype(int)
        assert np.array_equal(
            dataset.inequality_to_subsample("x_bool & y_bin", debug=debug),
            np.array((x > 5) & (y > 5)))

        # ~
        assert np.array_equal(
            dataset.inequality_to_subsample("~(x > 5)", debug=debug),
            np.array((x <= 5)))

        assert np.array_equal(
            dataset.inequality_to_subsample("(x > 5) & ~(y > 5)", debug=debug),
            np.array((x > 5) & ~(y > 5)))

        assert np.array_equal(
            dataset.inequality_to_subsample(
                "[(x > 5) | (y > 5)] & ~[(x > 5) & (y > 5)]", debug=debug),
            dataset.inequality_to_subsample(
                "[(x > 5) & (y <= 5)] | [(x <= 5) & (y > 5)]", debug=debug))

        assert np.array_equal(
            dataset.inequality_to_subsample("x_bool & ~y_bool", debug=debug),
            np.array((x > 5) & ~(y > 5)))

        # (10 > a > 5) will raise error
        # not a bug, we require the user to use [10 > a] & [a > 5]
        # () is for nonlogical operation, [] is for logical operation
        assert np.array_equal(
            dataset.inequality_to_subsample("[10 > x > 5]", debug=debug),
            np.array((x > 5) & (x < 10)))

    def test_parse(self):

        # parse_inequality
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

        # parse_and_or
        assert parse_and_or("[x > 3 & [ y > 5 | z < 2]]") == [
            '[', 'x>3', '&', '[', 'y>5', '|', 'z<2', ']', ']'
        ]

        # parse_op
        assert parse_op("(x + (1 + z / x))") == [
            '(', 'x', '+', '(', '1', '+', 'z', '/', 'x', ')', ')'
        ]


class TestLabels:

    def test_get_label_by_name(self):
        labels = Labels({
            'x': 'x label',
            'y': 'y label',
            'z': 'z label'
        },
                        units={'x': 'a'})

        assert labels.get('x') == 'x label a'
        assert labels.get('y') == 'y label'
        assert labels.get('x', with_unit=False) == 'x label'
        assert labels.get('log10@x') == r'$\log$x label a'

        assert labels.get('log10@x',
                          op_bracket='[{label}]') == r'$\log$[x label] a'
        assert labels.get('log10@x',
                          op_bracket='({label})') == r'$\log$(x label) a'

        labels.labels["log10@x"] = 'log10x'
        assert labels.get('log10@x', with_unit=False) == 'log10x'
