"""
ADV Kernel — Adaptive Density Variance Kernel
==============================================
A hybrid RBF × Polynomial kernel with locally-adaptive bandwidths
driven by k-NN density estimates and per-sample feature variance.

Quick start
-----------
>>> from adv_kernel import ADVKernelSVC
>>> model = ADVKernelSVC(C=1.0)
>>> model.fit(X_train, y_train)
>>> y_pred = model.predict(X_test)
"""

from .kernel import adv_bandwidth, adv_kernel
from .estimator import ADVKernelSVC

__all__ = ["adv_bandwidth", "adv_kernel", "ADVKernelSVC"]
__version__ = "0.1.0"
