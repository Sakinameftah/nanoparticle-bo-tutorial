"""The surrogate model: a Gaussian process (GP) with one length scale per variable.

kernel = C * Matern(length_scale=[l1, l2, l3], nu=2.5) + WhiteKernel(noise)

* C        how large the score variations are (vertical scale)
* l_i      how far variable i (in the 0-1 cube) must move before the score
           changes: a small l_i means variable i matters a lot ("ARD")
* noise    the part of the scatter the model attributes to experimental noise
* alpha    optional extra noise per experiment, e.g. for tubes flagged by a
           quality-control check; they then count less in the fit
"""

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

from .space import DIM


def make_gp(alpha=1e-6, n_restarts=5, seed=0,
            length_bounds=(0.05, 5.0), noise_bounds=(1e-5, 1.0)):
    kernel = (ConstantKernel(1.0, (1e-2, 1e2))
              * Matern(length_scale=[0.5] * DIM, length_scale_bounds=length_bounds, nu=2.5)
              + WhiteKernel(noise_level=0.05, noise_level_bounds=noise_bounds))
    return GaussianProcessRegressor(kernel=kernel, alpha=alpha, normalize_y=True,
                                    n_restarts_optimizer=n_restarts, random_state=seed)


def describe(gp):
    """Learned hyperparameters of a fitted GP, in plain numbers.

    The noise is returned both on the normalised scale used inside the GP and
    as a standard deviation in score units.
    """
    k = gp.kernel_
    amp = float(k.k1.k1.constant_value)
    lengths = np.atleast_1d(k.k1.k2.length_scale).astype(float)
    noise_norm = float(k.k2.noise_level)
    noise_sd = float(np.sqrt(noise_norm) * gp._y_train_std)
    return {"amplitude": amp, "length_scales": lengths,
            "noise_level_normalised": noise_norm, "noise_sd_score": noise_sd}


def fixed_kernel_gp(gp, alpha=1e-6):
    """A GP that reuses the learned kernel of `gp` without re-optimising it."""
    return GaussianProcessRegressor(kernel=gp.kernel_, optimizer=None,
                                    alpha=alpha, normalize_y=True)
