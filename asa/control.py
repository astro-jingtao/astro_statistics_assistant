import numpy as np
from scipy.stats import binned_statistic

from .utils import is_int

def control_1d(x_A,
               x_B,
               mode='match_P',
               dist_target=None,
               edges=None,
               bins=10,
               _range=None,
               P_atol=0,
               P_rtol=0):
    """
    Control the distribution of x_A and x_B to be the same.

    Parameters
    ----------
    x_A, x_B : array_like
        1D arrays.
    mode : str, optional
        'match_P' or 'match_N'. Default is 'match_P'.
    dist_target : array_like, optional
        1D array, target distribution. Default is None.
    edges : array_like, optional
        1D array, edges of bins. Default is None.
    bins : int, optional
        Number of bins, only used when both dist_target and edges are None. Default is 10.
    _range : list of 2 elements, optional
        Only used when both dist_target and edges are None. Default is None. 
        If _range is None, the minimum and maximum of the union set of x_A and x_B will be used.
    P_atol : float, optional
        Absolute tolerance for match_P, default is 0, only used when mode is 'match_P'.
    P_rtol : float, optional
        Relative tolerance for match_P, default is 0, only used when mode is 'match_P'.

    Returns
    -------
    A_index, B_index : array_like
        Indices of x_A and x_B after matching.
    dist_target : array_like
        Target distribution.
    edges : array_like
        Edges of bins.
    

    Raises
    ------
    ValueError
        If dist_target and edges are not both None or not None, or if mode is not 'match_N' or 'match_P'.

    Notes
    -----
    If dist_target and edges are both None, the dist_target will be the minimum of the histogram of x_A and x_B.
    """
    if (dist_target is None) and (edges is None):
        if _range is None:
            x_min = np.min(np.concatenate([x_A, x_B]))
            x_max = np.max(np.concatenate([x_A, x_B]))
            _range = [x_min, x_max]

        N_A, edges = np.histogram(x_A, density=False, bins=bins, range=_range)
        N_B, _ = np.histogram(x_B, density=False, bins=edges)
        dist_target = np.min([N_A, N_B], axis=0)

    elif (dist_target is not None) and (edges is not None):
        pass
    else:
        raise ValueError('dist_target and edges must be both None or not None')

    if mode == 'match_P':
        A_index = match_P_1d(x_A,
                             dist_target,
                             edges,
                             P_atol=P_atol,
                             P_rtol=P_rtol)
        B_index = match_P_1d(x_B,
                             dist_target,
                             edges,
                             P_atol=P_atol,
                             P_rtol=P_rtol)
    elif mode == 'match_N':
        A_index = match_N_1d(x_A, dist_target, edges)
        B_index = match_N_1d(x_B, dist_target, edges)

    else:
        raise ValueError('mode must be match_N or match_P')

    return A_index, B_index, dist_target, edges


def match_P_1d(x_parent, P_target, edges, P_atol=0, P_rtol=0):

    x_parent = np.asarray(x_parent)
    P_target = np.asarray(P_target)
    P_target = P_target / P_target.sum()

    N_parent, _ = np.histogram(x_parent, bins=edges, density=False)

    # N_max = N_parent / P_target
    # N_max[P_target == 0] = x_parent.size

    # from large to small
    # for this_N_max in np.sort(N_max)[::-1]:
    for this_N_max in np.arange(x_parent.size, 0, -1):
        N_can_have = np.min((N_parent, this_N_max * P_target),
                            axis=0).astype(int)
        if N_can_have.sum() <= 0:
            raise ValueError(
                'x_parent have no data in some of non-zero bins of P_target')
        P_can_have = N_can_have / N_can_have.sum()
        if np.allclose(P_can_have, P_target, atol=P_atol, rtol=P_rtol):
            N_dist_match = N_can_have
            break
    else:
        raise ValueError('No match found, try to increase P_atol or P_rtol')

    return match_N_1d(x_parent, N_dist_match, edges)


def match_N_1d(x_parent, N_target, edges):

    x_parent = np.asarray(x_parent)
    N_target = np.asarray(N_target)

    if not is_int(N_target):
        raise ValueError('N_target should contain only integers')

    edges = np.asarray(edges)

    _, _, bin_index = binned_statistic(x_parent,
                                       x_parent,
                                       statistic='count',
                                       bins=edges)
    # print(bin_index)
    index = np.arange(x_parent.size)
    index_selected = []
    for i in range(1, edges.size):
        in_this_bin = (bin_index == i)

        if N_target[i - 1] > in_this_bin.sum():
            right_backet = ']' if i == edges.size else ')'
            raise ValueError(
                f'x_parent have less data ({in_this_bin.sum()}) in the bin [{edges[i - 1]}, {edges[i]}{right_backet} than N_target requires ({N_target[i - 1]})'
            )

        if in_this_bin.any():
            # print(index[in_this_bin], N_target[i - 1], i)
            index_selected += list(
                np.random.choice(index[in_this_bin],
                                 size=N_target[i - 1],
                                 replace=False))

    # print(index_selected)
    return np.array(index_selected)
