import numpy as np
from scipy import linalg
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from .utils import remove_bad


def get_LDA_projection(x,
                       y,
                       n_components=1,
                       use_transform=True,
                       return_more=False,
                       LDA_kwargs=None):
    """Get the projection of the data onto the first n_components of LDA.

    Args:
        x (np.ndarray): The data to project.
        y (np.ndarray): The labels to use for training the LDA.
        n_components (int): The number of components to project onto.
        return_more (bool): Whether to return more information.

    Returns:
        np.ndarray: The projection of the data onto the first n_components of
            LDA.
    """
    if LDA_kwargs is None:
        LDA_kwargs = {}

    lda = LinearDiscriminantAnalysis(n_components=n_components, **LDA_kwargs)
    lda.fit(x, y)
    if use_transform:
        x_projected = lda.transform(x)
    else:
        x_projected = x @ lda.scalings_

    # sourcery skip: assign-if-exp
    if return_more:
        return x_projected, lda
    else:
        return x_projected

def lda_loop(X, y, n_loop=2):
    basics_proj = np.zeros((X.shape[1], n_loop))
    X_proj = X.copy()
    basics_null = np.eye(X.shape[1])
    for i in range(n_loop):
        lda = LinearDiscriminantAnalysis(n_components=1, solver='eigen')
        lda.fit(X_proj, y)
        lda_vec = lda.scalings_[:, 0:1]  # type: ignore
        lda_vec = lda_vec / np.linalg.norm(lda_vec)  # type: ignore
        basics_proj[:, i] = (basics_null @ lda_vec)[:, 0]
        # print(lda_null.shape, linalg.null_space(lda.scalings_[:, 0:1].T).shape)
        basics_null = basics_null @ linalg.null_space(lda_vec.T)
        X_proj = X @ basics_null
    return basics_proj, basics_null