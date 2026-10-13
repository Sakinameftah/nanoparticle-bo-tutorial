"""Choosing the next experiments: expected improvement and batch selection."""

import numpy as np
from scipy.stats import norm, qmc

from .model import fixed_kernel_gp
from .space import DIM


def expected_improvement(mu, sigma, best, xi=0.01):
    """EI = imp * Phi(z) + sigma * phi(z), with imp = mu - best - xi, z = imp / sigma.

    * first term: the predicted gain, weighted by the chance of beating `best`
      (exploitation);
    * second term: a bonus for uncertainty (exploration).
    """
    mu, sigma = np.asarray(mu, float), np.maximum(np.asarray(sigma, float), 1e-12)
    imp = mu - best - xi
    z = imp / sigma
    return imp * norm.cdf(z) + sigma * norm.pdf(z)


def incumbent(gp, U_obs, y_obs, kind="posterior"):
    """The 'best so far' that EI tries to beat.

    kind="observed"  : highest measured score (simple, but a lucky noisy
                       measurement sets the bar too high);
    kind="posterior" : highest *predicted* score among recipes already made
                       (more robust when measurements are noisy).
    """
    if kind == "observed":
        return float(np.max(y_obs))
    return float(np.max(gp.predict(U_obs)))


def kriging_believer(gp, U_obs, y_obs, n_batch, n_candidates=5000, xi=0.01,
                     min_dist=0.05, seed=0, best_kind="posterior", return_details=False):
    """Pick a batch of `n_batch` different recipes (unit cube) with EI.

    After each pick the GP 'believes' its own prediction there (it adds the
    point with value mu), so the uncertainty around that recipe collapses and
    the next pick goes elsewhere. The kernel learned on real data is frozen:
    the model never learns its hyperparameters from invented points.
    """
    cand = qmc.LatinHypercube(d=DIM, seed=seed).random(n_candidates)
    U_aug, y_aug = np.array(U_obs, float), np.array(y_obs, float)
    best = incumbent(gp, U_obs, y_obs, best_kind)
    gp_k, picks, details = gp, [], []
    for _ in range(n_batch):
        mu, sd = gp_k.predict(cand, return_std=True)
        ei = expected_improvement(mu, sd, best, xi)
        if picks:  # avoid near-duplicates of points already picked in this batch
            d = np.min(np.linalg.norm(cand[:, None, :] - np.array(picks)[None], axis=2), axis=1)
            ei[d < min_dist] = -np.inf
        j = int(np.argmax(ei))
        picks.append(cand[j])
        details.append({"mu": mu[j], "sigma": sd[j], "ei": ei[j]})
        U_aug = np.vstack([U_aug, cand[j]])
        y_aug = np.append(y_aug, mu[j])
        best = max(best, mu[j])  # a believed value can become the new "best so far"
        cand = np.delete(cand, j, axis=0)
        gp_k = fixed_kernel_gp(gp).fit(U_aug, y_aug)
    picks = np.array(picks)
    return (picks, details) if return_details else picks
