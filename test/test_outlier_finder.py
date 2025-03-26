import numpy as np

from asa.outlier_finder import find_contour_outliers


class TestFindContourOutliers:

    def test_normal(self):
        # Test with a high number of points near each other
        N = 10_000
        x = np.random.normal(loc=0, scale=1, size=N)
        y = np.random.normal(loc=0, scale=1, size=N)
        level = 0.8
        result, checker = find_contour_outliers(x, y, level)

        # Check that result is still a numpy array
        assert isinstance(result, np.ndarray)
        assert result.shape == x.shape  # Should match input shape

        assert checker.is_inside([0], [0])
        assert not checker.is_inside([10], [10])
