"""Running whole campaigns on the virtual lab, and comparing strategies fairly."""

import numpy as np
import pandas as pd

from .acquisition import kriging_believer
from .design import latin_hypercube, random_design
from .model import make_gp
from .scores import desirability_score
from .space import DIM, SHORT, centre_point, from_unit, to_unit
from .virtual_lab import VirtualLab, true_properties


def _rows(X, size, disp, rnd, reason, score_fn):
    df = pd.DataFrame(X, columns=SHORT)
    df.insert(0, "round", rnd)
    df.insert(1, "reason", reason)
    df["size_nm"], df["disp_pct"] = size, disp
    df["score"] = score_fn(size, disp)
    ts, td = true_properties(X)
    df["true_score"] = score_fn(ts, td)  # only known because the lab is virtual
    return df


def run_bo(n_init=6, batch=6, n_rounds=4, n_centre=0, lab=None, seed=0,
           score_fn=desirability_score, n_restarts=3, n_candidates=4000, xi=0.01):
    """Latin hypercube first, then `n_rounds` batches chosen by EI + kriging believer.

    `n_centre` repeats of the centre point are added to every round; they cost
    experiments but let the model measure the noise.
    """
    lab = lab or VirtualLab(seed=seed)
    parts = []
    X0 = latin_hypercube(n_init, seed=seed)
    parts.append(_rows(X0, *lab.run(X0), 0, "LHS", score_fn))
    if n_centre:
        Xc = np.repeat(centre_point(), n_centre, axis=0)
        parts.append(_rows(Xc, *lab.run(Xc), 0, "centre", score_fn))
    for rnd in range(1, n_rounds + 1):
        hist = pd.concat(parts, ignore_index=True)
        U, y = to_unit(hist[SHORT].to_numpy()), hist["score"].to_numpy()
        gp = make_gp(n_restarts=n_restarts, seed=seed).fit(U, y)
        n_model = batch - n_centre
        Xn = from_unit(kriging_believer(gp, U, y, n_model, n_candidates=n_candidates,
                                        xi=xi, seed=seed * 100 + rnd))
        parts.append(_rows(Xn, *lab.run(Xn), rnd, "model", score_fn))
        if n_centre:
            Xc = np.repeat(centre_point(), n_centre, axis=0)
            parts.append(_rows(Xc, *lab.run(Xc), rnd, "centre", score_fn))
    return pd.concat(parts, ignore_index=True)


def run_design(kind, n_total, lab=None, seed=0, score_fn=desirability_score):
    """A model-free design of `n_total` experiments: 'lhs' or 'random'."""
    lab = lab or VirtualLab(seed=seed)
    X = latin_hypercube(n_total, seed=seed) if kind == "lhs" else random_design(n_total, seed=seed)
    return _rows(X, *lab.run(X), 0, kind.upper(), score_fn)


def run_ofat(n_total, order=(0, 1, 2), lab=None, seed=0, score_fn=desirability_score):
    """One factor at a time: scan one variable (others fixed), keep the best, move on."""
    lab = lab or VirtualLab(seed=seed)
    per_var = n_total // DIM
    x = np.full(DIM, 0.5)
    parts = []
    for step, k in enumerate(order):
        U = np.tile(x, (per_var, 1))
        U[:, k] = np.linspace(0, 1, per_var)
        X = from_unit(U)
        df = _rows(X, *lab.run(X), step, f"OFAT {SHORT[k]}", score_fn)
        parts.append(df)
        x = U[int(np.argmax(df["score"]))]
    return pd.concat(parts, ignore_index=True)


def best_so_far(df):
    """After n experiments: TRUE score of the recipe with the best MEASURED score.

    This is what a chemist would actually take home, so a lucky noisy
    measurement is not rewarded.
    """
    meas, true = df["score"].to_numpy(), df["true_score"].to_numpy()
    out, best_i = [], 0
    for n in range(len(df)):
        if meas[n] > meas[best_i]:
            best_i = n
        out.append(true[best_i])
    return np.array(out)
