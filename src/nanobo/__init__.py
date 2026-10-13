"""nanobo: a minimal, readable Bayesian-optimisation loop for nanoparticle synthesis.

Teaching code. The `virtual_lab` module is an invented response surface,
not a model of real chemistry.
"""

from .space import VARIABLES, NAMES, SHORT, BOUNDS, DIM, to_unit, from_unit, centre_point
from .design import latin_hypercube, random_design, slices_used
from .virtual_lab import VirtualLab, true_properties
from .scores import penalty_score, desirability_score, d_size, d_disp
from .model import make_gp, describe, fixed_kernel_gp
from .acquisition import expected_improvement, incumbent, kriging_believer
from .campaign import run_bo, run_design, run_ofat, best_so_far

__version__ = "0.1.0"
