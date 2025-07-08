import numpy as np
from sklearn.mixture import GaussianMixture as SK_GaussianMixture

from asa.sklearn_tools.mixture import GaussianMixture


class TestGaussianMixture:

    def test_downward_compatibility(self):
        # should work as scikit-learn's GaussianMixture
        # if no new feature used
        X = np.random.normal(size=(100, 2))
        gmm = GaussianMixture(n_components=2,
                              covariance_type='full',
                              random_state=0)
        gmm_sk = SK_GaussianMixture(n_components=2,
                                    covariance_type='full',
                                    random_state=0)

        gmm.fit(X)
        gmm_sk.fit(X)

        assert np.allclose(gmm.means_, gmm_sk.means_)
        assert np.allclose(gmm.covariances_, gmm_sk.covariances_)
        assert np.allclose(gmm.weights_, gmm_sk.weights_)
        assert np.allclose(gmm.predict(X), gmm_sk.predict(X))
        assert np.allclose(gmm.score(X), gmm_sk.score(X))

    def test_fix_means(self):
        X = np.random.normal(size=(100, 2))
        gmm = GaussianMixture(n_components=2,
                              covariance_type='full',
                              random_state=0,
                              fix_means=True,
                              means_init=np.array([[0, 0], [1, 1]]))

        gmm.fit(X)
        assert np.allclose(gmm.means_, np.array([[0, 0], [1, 1]]))

    def test_fix_weights(self):
        X = np.random.normal(size=(100, 2))
        gmm = GaussianMixture(n_components=2,
                              covariance_type='full',
                              random_state=0,
                              fix_weights=True,
                              weights_init=np.array([0.6, 0.4]))

        gmm.fit(X)
        assert np.allclose(gmm.weights_, np.array([0.6, 0.4]))
