"""Experimental designs that need no model: Latin hypercube, random, one factor at a time."""

import numpy as np
from scipy.stats import qmc

from .space import DIM, from_unit


def latin_hypercube(n, seed=0, unit=False):
    """n recipes that use each of n slices of every variable exactly once."""
    U = qmc.LatinHypercube(d=DIM, seed=seed).random(n)
    return U if unit else from_unit(U)


def random_design(n, seed=0, unit=False):
    """n recipes drawn uniformly at random (no stratification)."""
    U = np.random.default_rng(seed).random((n, DIM))
    return U if unit else from_unit(U)


def slices_used(U, n=None):
    """For each variable, the slice index (0..n-1) of each point.

    For a Latin hypercube every row of the result is a permutation of 0..n-1.
    """
    U = np.atleast_2d(U)
    n = n or len(U)
    return np.minimum((U * n).astype(int), n - 1).T
