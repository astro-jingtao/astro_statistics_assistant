import numpy as np

from asa.robust_statistic import sigma_clip
from asa.weighted_statistic import quantile


class TestSigmaClip:

    def test_basic(self):

        data = np.array([1, 2, 3, 4, 1000])
        m, s, is_good = sigma_clip(data)
        assert np.all(is_good == [True, True, True, True, False])
        assert m == np.median([1, 2, 3, 4])
        assert s == (quantile([1, 2, 3, 4], q=0.84) -
                     quantile([1, 2, 3, 4], q=0.16)) / 2
