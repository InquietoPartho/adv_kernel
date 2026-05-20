"""
adv_kernel.estimator
--------------------
scikit-learn compatible SVC wrapper that uses the ADV kernel.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.svm import SVC
from sklearn.utils.validation import check_is_fitted

from .kernel import adv_bandwidth, adv_kernel


class ADVKernelSVC(BaseEstimator, ClassifierMixin):
    """Support Vector Classifier with the Adaptive Density Variance kernel.

    Parameters
    ----------
    C : float, default=1.0
        Regularisation parameter.
    beta : float, default=0.5
        Scaling factor for the polynomial term.
    degree : int, default=2
        Degree of the polynomial term.
    k_bw : int, default=10
        Number of nearest neighbours for bandwidth estimation.
    gamma_density : float, default=1.0
        Weight of the k-NN density component.
    gamma_var : float, default=0.5
        Weight of the feature-variance component.
    probability : bool, default=True
        Whether to enable predict_proba.

    Examples
    --------
    >>> import numpy as np
    >>> from adv_kernel import ADVKernelSVC
    >>> rng = np.random.default_rng(0)
    >>> X = rng.standard_normal((100, 5))
    >>> y = (X[:, 0] > 0).astype(int)
    >>> clf = ADVKernelSVC(C=1.0, probability=False)
    >>> clf.fit(X, y)
    ADVKernelSVC(probability=False)
    """

    def __init__(
        self,
        C: float = 1.0,
        beta: float = 0.5,
        degree: int = 2,
        k_bw: int = 10,
        gamma_density: float = 1.0,
        gamma_var: float = 0.5,
        probability: bool = True,
    ) -> None:
        self.C = C
        self.beta = beta
        self.degree = degree
        self.k_bw = k_bw
        self.gamma_density = gamma_density
        self.gamma_var = gamma_var
        self.probability = probability

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ADVKernelSVC":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.X_train_ = X
        self.sigma_train_ = adv_bandwidth(X, self.k_bw, self.gamma_density, self.gamma_var)
        K_train = adv_kernel(X, X, self.sigma_train_, self.sigma_train_, self.beta, self.degree)
        self.svc_ = SVC(kernel="precomputed", C=self.C, probability=self.probability)
        self.svc_.fit(K_train, y)
        self.classes_ = self.svc_.classes_
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self)
        return self.svc_.predict(self._kernel_test(X))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self)
        return self.svc_.predict_proba(self._kernel_test(X))

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self)
        return self.svc_.decision_function(self._kernel_test(X))

    def _kernel_test(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        sigma_test = adv_bandwidth(X, self.k_bw, self.gamma_density, self.gamma_var)
        return adv_kernel(X, self.X_train_, sigma_test, self.sigma_train_, self.beta, self.degree)
