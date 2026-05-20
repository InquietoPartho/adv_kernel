"""Tests for adv-kernel."""

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score

from adv_kernel import adv_bandwidth, adv_kernel, ADVKernelSVC


@pytest.fixture
def small_data():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((60, 6))
    y = (X[:, 0] > 0).astype(int)
    return X, y


def test_bandwidth_shape(small_data):
    X, _ = small_data
    assert adv_bandwidth(X).shape == (X.shape[0],)


def test_bandwidth_positive(small_data):
    X, _ = small_data
    assert np.all(adv_bandwidth(X) > 0)


def test_bandwidth_gamma_density_zero(small_data):
    X, _ = small_data
    sigma = adv_bandwidth(X, gamma_density=0.0, gamma_var=1.0)
    np.testing.assert_allclose(sigma, np.var(X, axis=1) + 1e-12, rtol=1e-6)


def test_kernel_shape(small_data):
    X, _ = small_data
    sigma = adv_bandwidth(X)
    assert adv_kernel(X, X, sigma, sigma).shape == (len(X), len(X))


def test_kernel_symmetric(small_data):
    X, _ = small_data
    sigma = adv_bandwidth(X)
    K = adv_kernel(X, X, sigma, sigma)
    np.testing.assert_allclose(K, K.T, atol=1e-10)


def test_kernel_rectangular(small_data):
    X, _ = small_data
    Xtr, Xte = X[:40], X[40:]
    K = adv_kernel(Xte, Xtr, adv_bandwidth(Xte), adv_bandwidth(Xtr))
    assert K.shape == (20, 40)


def test_fit_predict(small_data):
    X, y = small_data
    clf = ADVKernelSVC(probability=False).fit(X, y)
    preds = clf.predict(X)
    assert preds.shape == (len(X),)
    assert set(preds).issubset({0, 1})


def test_predict_proba(small_data):
    X, y = small_data
    proba = ADVKernelSVC().fit(X, y).predict_proba(X)
    assert proba.shape == (len(X), 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)


def test_decision_function(small_data):
    X, y = small_data
    scores = ADVKernelSVC(probability=False).fit(X, y).decision_function(X)
    assert scores.shape == (len(X),)


def test_classes_attribute(small_data):
    X, y = small_data
    clf = ADVKernelSVC().fit(X, y)
    np.testing.assert_array_equal(clf.classes_, [0, 1])


def test_sklearn_cross_val(small_data):
    X, y = small_data
    scores = cross_val_score(ADVKernelSVC(probability=False), X, y, cv=3)
    assert all(0 <= s <= 1 for s in scores)


def test_multiclass():
    X, y = make_classification(
        n_samples=90, n_features=4, n_classes=3,
        n_informative=3, n_redundant=0, random_state=7
    )
    clf = ADVKernelSVC(probability=False).fit(X, y)
    assert set(clf.predict(X)).issubset({0, 1, 2})
