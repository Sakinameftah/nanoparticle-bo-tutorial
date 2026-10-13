"""Turning several measured properties into one number.

Two scores are provided:

* `penalty_score`: 100 minus weighted squared penalties. Easy to read, but it
  has no floor (one failed synthesis can score -10 000 and dominate the
  model) and it mixes units through hand-picked weights.
* `desirability_score`: each property is mapped to a desirability between 0
  and 1, and the score is their geometric mean. It is bounded, unit-free,
  and a single unacceptable property gives a zero score (no compensation).
"""

import numpy as np

TARGET_SIZE = 10.0  # nm
MAX_GOOD_DISP = 4.0  # %


def penalty_score(size, disp, w_size=5.0, w_disp=15.0):
    """100 - w_size*(size-10)^2 - w_disp*max(0, disp-4)^2."""
    size, disp = np.asarray(size, float), np.asarray(disp, float)
    return (100.0 - w_size * (size - TARGET_SIZE) ** 2
            - w_disp * np.maximum(0.0, disp - MAX_GOOD_DISP) ** 2)


def d_size(size, tol=1.5):
    """1 at the target size, exp(-1) = 0.37 at target +/- tol."""
    return np.exp(-(((np.asarray(size, float) - TARGET_SIZE) / tol) ** 2))


def d_disp(disp, worst=12.0):
    """1 for dispersion <= 4 %, falling linearly to 0 at `worst` %."""
    disp = np.asarray(disp, float)
    return np.clip((worst - disp) / (worst - MAX_GOOD_DISP), 0.0, 1.0)


def desirability_score(size, disp):
    """Geometric mean of the two desirabilities, between 0 and 1."""
    return np.sqrt(d_size(size) * d_disp(disp))
