# ADV Kernel — Adaptive Density Variance Kernel

[![CI](https://github.com/InquietoPartho/adv_kernel/actions/workflows/ci.yml/badge.svg)](https://github.com/InquietoPartho/adv_kernel/actions)
[![PyPI](https://img.shields.io/pypi/v/adv-kernel)](https://pypi.org/project/adv-kernel/)
[![Python](https://img.shields.io/pypi/pyversions/adv-kernel)](https://pypi.org/project/adv-kernel/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

A scikit-learn compatible **SVM kernel** that adapts its bandwidth sample-by-sample using two complementary signals:

| Signal | Meaning |
|---|---|
| **k-NN density** | Sparse neighbourhood → wider kernel |
| **Feature variance** | High intra-sample heterogeneity → wider kernel |

The final kernel is a **pointwise product** of an adaptive RBF and a polynomial term:

```
K(xᵢ, xⱼ) = exp(−‖xᵢ−xⱼ‖² / 2σᵢσⱼ) × (1 + β⟨xᵢ,xⱼ⟩)^degree
```

---

## Installation

```bash
pip install adv-kernel
```

Or directly from GitHub:

```bash
pip install git+https://github.com/InquietoPartho/adv_kernel.git
```

---

## Quick start

```python
from adv_kernel import ADVKernelSVC

clf = ADVKernelSVC(C=1.0, beta=0.5, degree=2, probability=True)
clf.fit(X_train, y_train)

y_pred  = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)
```

### Use kernel functions directly

```python
from adv_kernel import adv_bandwidth, adv_kernel

sigma = adv_bandwidth(X, k=10, gamma_density=1.0, gamma_var=0.5)
K     = adv_kernel(X, X, sigma, sigma, beta=0.5, degree=2)
```

### scikit-learn GridSearchCV

```python
from sklearn.model_selection import GridSearchCV
from adv_kernel import ADVKernelSVC

param_grid = {"C": [0.1, 1.0, 10.0], "beta": [0.2, 0.5], "gamma_density": [0.5, 1.0]}
gs = GridSearchCV(ADVKernelSVC(), param_grid, cv=5, n_jobs=-1)
gs.fit(X_train, y_train)
print(gs.best_params_)
```

---

## API reference

### `adv_bandwidth(X, k=10, gamma_density=1.0, gamma_var=0.5)`
Returns per-sample bandwidth array of shape `(n_samples,)`.

### `adv_kernel(X, Y, sigma_X, sigma_Y, beta=0.5, degree=2)`
Returns kernel matrix of shape `(n_X, n_Y)`.

### `ADVKernelSVC` parameters

| Parameter | Default | Description |
|---|---|---|
| `C` | `1.0` | SVM regularisation |
| `beta` | `0.5` | Polynomial scaling |
| `degree` | `2` | Polynomial degree |
| `k_bw` | `10` | k-NN neighbours for bandwidth |
| `gamma_density` | `1.0` | Density term weight |
| `gamma_var` | `0.5` | Variance term weight |
| `probability` | `True` | Enable `predict_proba` |

---

## License

MIT © Pijush Kanti Roy Partho
