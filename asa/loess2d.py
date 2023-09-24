import numpy as np
from scipy.spatial import KDTree


class LOESS2D:
    def __init__(self, x, y, val, n_nbr, boxsize=None, xy_ratio=1):
        """
        x, y, val: ndarray with shape (N, )
            input coordinate and values
        n_nbr: number of neighbours for smoothing, or fraction w.r.t. to the total population
            # of neighbors = (n_nbr >= 1) ? int(n_nbr) : int(n_nbr * N)
        boxsize: optional
            if assigned a value, the distance is calculated in a periodic box
        xy_ratio:
            weight in the calculation of distance
            d = sqrt(xy_ratio * (x_1 - x_2)^2 + (y_1 - y_2)^2)^2
        """
        # Record the transformation for x and y coordinates
        self._xnorm = self._gen_norm(x, xy_ratio)
        self._ynorm = self._gen_norm(y)
        self._xn = self._xnorm(x)
        self._yn = self._ynorm(y)
        self._val = val.copy()
        self._tree = KDTree(
            np.column_stack((self._xn, self._yn)), copy_data=True, boxsize=boxsize
        )
        self._n_nbr = int(n_nbr) if n_nbr >= 1 else int(n_nbr * len(x))
        if self._n_nbr > len(x):
            raise ValueError(
                "Number of smoothing neighbors exceeds the total number of points"
            )
        print("# of neightbours for smoothing: %d" % self._n_nbr)

    def __call__(self, x, y):
        x_norm = self._xnorm(x)
        y_norm = self._ynorm(y)
        d_nbr, i_nbr = self._tree.query(np.column_stack((x_norm, y_norm)), self._n_nbr)
        d_norm = (d_nbr.T / d_nbr[:, -1]).T
        weight = np.power(1 - np.power(d_norm, 3), 3)
        return np.sum(weight * self._val[i_nbr], axis=1) / np.sum(weight, axis=1)

    def _gen_norm(self, arr, ratio=1):
        """
        Normalize the coordinate using quantiles rather than the standard deviation
        to avoid the impact of outliners.
        """
        xl, x_med, xu = np.quantile(arr, [0.17, 0.5, 0.84])
        return lambda x: (x - x_med) / (xu - xl) * ratio