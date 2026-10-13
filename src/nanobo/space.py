"""The design space: three synthesis variables and their ranges.

Every model in this package works in the unit cube [0, 1]^3, so that a
concentration in mol/L, a molar ratio and a temperature in degrees Celsius
weigh the same. `to_unit` and `from_unit` convert between the two.
"""

import numpy as np

# name, unit, low, high
VARIABLES = [
    ("Fe-oleate concentration", "M", 0.05, 0.30),
    ("Oleic acid : Fe ratio", "", 1.0, 5.0),
    ("Reflux temperature", "°C", 280.0, 320.0),
]

NAMES = [v[0] for v in VARIABLES]
SHORT = ["conc_M", "ratio_OA_Fe", "temp_C"]
BOUNDS = np.array([[v[2], v[3]] for v in VARIABLES], dtype=float)
DIM = len(VARIABLES)


def to_unit(X):
    """Real units -> unit cube."""
    X = np.atleast_2d(np.asarray(X, dtype=float))
    return (X - BOUNDS[:, 0]) / (BOUNDS[:, 1] - BOUNDS[:, 0])


def from_unit(U):
    """Unit cube -> real units."""
    U = np.atleast_2d(np.asarray(U, dtype=float))
    return BOUNDS[:, 0] + U * (BOUNDS[:, 1] - BOUNDS[:, 0])


def centre_point():
    """The middle of the design space (used for repeated 'centre point' runs)."""
    return from_unit(np.full((1, DIM), 0.5))
