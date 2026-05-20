"""
adv_kernel.kernel
-----------------
Pure-function layer: bandwidth estimation and kernel evaluation.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics.pairwise import euclidean_distances


def adv_bandwidth(
    X: np.ndarray,
    k: int = 10,
    gamma_density: float = 1.0,
    gamma_var: float = 0.5,
) -> np.ndarray:
    """Compute per-sample adaptive bandwidths.

    Parameters
    ----------
    X : array of shape (n_samples, n_features)
    k : int, default=10
        Number of nearest neighbours for the density estimate.
    gamma_density : float, default=1.0
        Weight applied to the k-NN density term.
    gamma_var : float, default=0.5
        Weight applied to the per-sample feature-variance term.

    Returns
    -------
    sigma : ndarray of shape (n_samples,)
    """
    X = np.asarray(X, dtype=float)
    D = euclidean_distances(X, X)
    np.fill_diagonal(D, np.inf)
    knn_med = np.median(np.sort(D, axis=1)[:, :k], axis=1)
    feature_var = np.var(X, axis=1)
    return gamma_density * knn_med + gamma_var * feature_var + 1e-12


def adv_kernel(
    X: np.ndarray,
    Y: np.ndarray,
    sigma_X: np.ndarray,
    sigma_Y: np.ndarray,
    beta: float = 0.5,
    degree: int = 2,
) -> np.ndarray:
    """Evaluate the Adaptive Density Variance (ADV) kernel matrix.

    K(x_i, y_j) = exp(-||x_i - y_j||^2 / (2 * sigma_i * sigma_j))
                  * (1 + beta * <x_i, y_j>)^degree

    Parameters
    ----------
    X : array of shape (n_X, n_features)
    Y : array of shape (n_Y, n_features)
    sigma_X : array of shape (n_X,)
    sigma_Y : array of shape (n_Y,)
    beta : float, default=0.5
    degree : int, default=2

    Returns
    -------
    K : ndarray of shape (n_X, n_Y)
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    sigma_X = np.asarray(sigma_X, dtype=float)
    sigma_Y = np.asarray(sigma_Y, dtype=float)

    D2 = euclidean_distances(X, Y, squared=True)
    B = np.outer(sigma_X, sigma_Y)
    K_rbf = np.exp(-D2 / (2.0 * B))
    K_poly = (1.0 + beta * (X @ Y.T)) ** degree
    return K_rbf * K_poly
