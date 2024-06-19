import pytest
import numpy as np
from asa.control import control_1d, match_P_1d, match_N_1d

np.random.seed(0)


class Test1D:
    def test_control_1d_explicit(self):
        x_A = np.array([1, 1, 2, 2, 2, 3])
        x_B = np.array([1, 2, 2, 2, 3, 3])
        dist_target = [1, 1]
        edges = [1, 1.5, 3]

        # test mode='match_P'
        A_index, B_index, _dist_target, _edges = control_1d(
            x_A, x_B, mode='match_P', dist_target=dist_target, edges=edges)
        assert np.array_equal(dist_target, _dist_target)
        assert np.array_equal(edges, _edges)
        assert np.array_equal(
            np.histogram(x_A[A_index], bins=edges)[0], [2, 2])
        assert np.array_equal(
            np.histogram(x_B[B_index], bins=edges)[0], [1, 1])

        # test mode='match_N'
        A_index, B_index, _dist_target, _edges = control_1d(
            x_A, x_B, mode='match_N', dist_target=dist_target, edges=edges)
        assert np.array_equal(dist_target, _dist_target)
        assert np.array_equal(edges, _edges)
        assert np.array_equal(
            np.histogram(x_A[A_index], bins=edges)[0], [1, 1])
        assert np.array_equal(
            np.histogram(x_B[B_index], bins=edges)[0], [1, 1])

    def test_control_1d_auto(self):

        x_A = np.array([1, 1, 2, 2, 2, 3])
        x_B = np.array([1, 2, 2, 2, 3, 3])

        A_index, B_index, dist_target, edges = control_1d(x_A,
                                                          x_B,
                                                          mode='match_N',
                                                          bins=2)

        assert np.array_equal(
            np.histogram(x_A[A_index], bins=edges)[0], [1, 4])
        assert np.array_equal(
            np.histogram(x_B[B_index], bins=edges)[0], [1, 4])

    def test_match_P_1d(self):
        x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        P_target = [1, 1, 1]
        edges = [0, 3, 6, 10]

        index = match_P_1d(x, P_target, edges)
        assert np.array_equal(np.histogram(x[index], bins=edges)[0], [3, 3, 3])

        x = np.random.randn(100)
        P_target = np.array([1, 2, 1])
        edges = [-3, -1, 1, 3]

        index = match_P_1d(x, P_target, edges)
        P_control = np.histogram(x[index], bins=edges)[0]
        assert np.allclose(P_control / P_control.sum(),
                           P_target / P_target.sum())

        x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        P_target = [1, 1, 1]
        edges = [0, 3, 6, 10]

        index = match_P_1d(x, P_target, edges)
        assert np.array_equal(np.histogram(x[index], bins=edges)[0], [3, 3, 3])

        x = np.array([2] * 10 + [5] * 10 + [8] * 10)
        P_target = [1, 3, 1]
        edges = [0, 3, 6, 10]

        index = match_P_1d(x, P_target, edges)
        assert np.array_equal(np.histogram(x[index], bins=edges)[0], [3, 9, 3])

    def test_match_N_1d(self):

        x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        N_target = [2, 1, 2]
        edges = [0, 3, 6, 10]

        index = match_N_1d(x, N_target, edges)
        # print(x[index])
        assert np.array_equal(np.histogram(x[index], bins=edges)[0], N_target)

        N_target = [10, 1, 2]
        edges = [0, 3, 6, 10]

        with pytest.raises(ValueError) as excinfo:
            index = match_N_1d(x, N_target, edges)
        assert "x_parent have less data (3) in the bin [0, 3) than N_target requires (10)" == str(
            excinfo.value)

        x = np.random.randn(100)
        N_target = [5, 10, 5]
        edges = [-3, -1, 1, 3]

        index = match_N_1d(x, N_target, edges)
        assert np.array_equal(np.histogram(x[index], bins=edges)[0], N_target)
