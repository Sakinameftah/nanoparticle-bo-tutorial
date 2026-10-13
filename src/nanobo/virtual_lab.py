"""A virtual laboratory: an INVENTED response surface plus measurement noise.

Nothing here is fitted to real data. The equations are a toy model that only
mimics a few qualitative features of the thermal decomposition of iron
oleate, chosen for teaching:

* size grows with temperature and concentration and shrinks with more ligand;
* the size distribution is narrowest along a *diagonal* ridge where the
  ligand ratio has to increase with temperature, so the variables interact
  and one-factor-at-a-time searches tend to stop early;
* every "synthesis" returns noisy measurements, like a real TEM/DLS series.

Use it to test optimisation strategies, never to draw chemical conclusions.
"""

import numpy as np

from .space import to_unit

# Where the (hidden) best recipe sits, in unit-cube coordinates.
_OPT = np.array([0.75, 0.85, 0.80])


def true_properties(X):
    """Noise-free size (nm) and dispersion (%) for recipes X in real units."""
    u = to_unit(X)
    c, r, t = (u - _OPT).T
    size = 10 + 16 * t + 6 * c - 9 * r + 20 * c * t
    r_ridge = 0.9 * t  # the best ligand ratio moves with temperature
    disp = 2.5 + 200 * (r - r_ridge) ** 2 + 12 * c**2 + 10 * t**2
    return np.clip(size, 1.0, None), np.clip(disp, 1.0, None)


class VirtualLab:
    """Runs 'syntheses' and returns noisy measurements.

    Parameters
    ----------
    size_noise : float
        Standard deviation of the measured size, in nm.
    disp_noise : float
        Standard deviation of the measured dispersion, in percentage points.
    batch_sd : float
        Optional hidden batch effect (in nm) shared by all syntheses of a
        round, e.g. a new precursor lot. 0 by default.
    seed : int
        Random seed, so that a campaign can be replayed exactly.
    """

    def __init__(self, size_noise=0.5, disp_noise=0.8, batch_sd=0.0, seed=0):
        self.size_noise = size_noise
        self.disp_noise = disp_noise
        self.batch_sd = batch_sd
        self.rng = np.random.default_rng(seed)
        self.n_runs = 0

    def run(self, X):
        """Synthesise and measure the recipes X (n x 3, real units)."""
        X = np.atleast_2d(X)
        size, disp = true_properties(X)
        offset = self.rng.normal(0, self.batch_sd) if self.batch_sd > 0 else 0.0
        size = size + offset + self.rng.normal(0, self.size_noise, len(X))
        disp = disp + self.rng.normal(0, self.disp_noise, len(X))
        self.n_runs += len(X)
        return np.clip(size, 0.5, None), np.clip(disp, 0.5, None)
