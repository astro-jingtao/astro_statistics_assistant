import numpy as np
import matplotlib.pyplot as plt


def imshow(X, *, ax=None, mask=None, pmin=None, pmax=None, **kwargs):
    if ax is None:
        ax = plt.gca()

    if mask is not None:
        X = X.astype(float).copy()
        X[~mask] = np.nan

    if pmin is not None:
        if 'vmin' in kwargs:
            print("pmin is ignored because vmin is provided")
        else:
            kwargs['vmin'] = np.nanpercentile(X, pmin)

    if pmax is not None:
        if 'vmax' in kwargs:
            print("pmax is ignored because vmax is provided")
        else:
            kwargs['vmax'] = np.nanpercentile(X, pmax)

    return ax.imshow(X, **kwargs)
