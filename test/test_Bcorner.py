import numpy as np
import pytest
from asa.Bcorner import quantile, corner, hist2d


class TestQuantile:

    def test_quantile(self):
        x = np.random.normal(size=1000)
        assert np.isclose(quantile(x, 0.5)[0], np.median(x))
        assert quantile(list(x), 0.5)[0] == quantile(x, 0.5)[0]
        assert np.isnan(quantile([], 0.5)[0])

    def test_smooth_confict(self):
        # should rasie error

        with pytest.raises(ValueError) as excinfo:
            corner(np.random.normal(size=(1000, 2)),
                   smooth1d=1,
                   kde_smooth1d=True)
        assert "kde_smooth1d and smooth1d cannot be set at the same time" == str(
            excinfo.value)

        with pytest.raises(ValueError) as excinfo:
            hist2d(np.random.normal(size=1000),
                   np.random.normal(size=1000),
                   smooth=1,
                   kde_smooth=True)
        assert "kde_smooth and smooth cannot be set at the same time" == str(
            excinfo.value)
