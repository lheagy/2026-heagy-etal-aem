"""Sensitivity weights for a tiled TDEM simulation.

`directives.UpdateSensitivityWeights` needs `sim.getJtJdiag`, which
`Simulation3DElectricField` does not implement (FDEM, DC, IP, NSEM-1D, gravity
and magnetics do; time domain does not), and `MetaSimulation.getJtJdiag` only
delegates to the sub-simulations. `Jtvec` *is* available, so estimate the
diagonal by Hutchinson probing.

For the data-weighted Jacobian `A = W J` (W diagonal = 1/std) and Rademacher
`v` (+/-1, length n_d):

    (A^T v)_j    = sum_i W_ii J_ij v_i = [J^T (W v)]_j
    E[(A^T v)_j^2] = sum_i W_ii^2 J_ij^2 = diag(A^T A)_j

so the mean of `Jtvec(m, W_diag * v)**2` over probes is an unbiased estimate of
`diag(J^T W^T W J)` -- exactly what `UpdateSensitivityWeights` computes.

`sensitivity_weights` then reproduces that directive's convention (read off
`UpdateSensitivityWeights.update`): sqrt(jtj/vol**2), amplitude threshold,
normalize by the maximum.
"""
import time

import numpy as np


def estimate_jtj_diag(sim, m, W_diag, n_probe=32, seed=0, verbose=True,
                      reuse_fields=True):
    """Hutchinson estimate of diag((W J)^T (W J)) using only Jtvec.

    Parameters
    ----------
    sim : simulation with a `Jtvec(m, v, f=None)` method
    m : (n_param,) model to linearize about
    W_diag : (n_data,) diagonal of the data weighting matrix, i.e. 1/std
    n_probe : number of Rademacher probes; variance falls as 1/n_probe
    seed : RNG seed, so the weights are reproducible
    reuse_fields : compute `sim.fields(m)` once and pass it to every Jtvec

    Returns
    -------
    (n_param,) estimate of the weighted JtJ diagonal
    """
    W_diag = np.asarray(W_diag, dtype=float)
    rng = np.random.default_rng(seed)

    f = None
    if reuse_fields:
        try:
            t0 = time.perf_counter()
            f = sim.fields(m)
            if verbose:
                print(f"  fields for probing: {time.perf_counter()-t0:.1f} s",
                      flush=True)
        except Exception as err:            # noqa: BLE001 - fall back, not fatal
            print(f"  could not cache fields ({type(err).__name__}); "
                  f"each probe will recompute them", flush=True)
            f = None

    acc = None
    t_start = time.perf_counter()
    for k in range(n_probe):
        v = rng.integers(0, 2, size=W_diag.size).astype(float) * 2.0 - 1.0
        try:
            jt = sim.Jtvec(m, W_diag * v, f=f)
        except TypeError:                   # signature without f
            jt = sim.Jtvec(m, W_diag * v)
        jt = np.asarray(jt, dtype=float)
        acc = jt**2 if acc is None else acc + jt**2
        if verbose and (k + 1) % 8 == 0:
            el = time.perf_counter() - t_start
            print(f"  probe {k+1}/{n_probe}  ({el:.0f} s elapsed, "
                  f"{el/(k+1):.1f} s/probe)", flush=True)

    return acc / n_probe


def exact_jtj_diag(sim, m, W_diag, verbose=True):
    """Exact diag((W J)^T (W J)) by building every row of J.

    Only affordable when n_data is small (one sounding = 20 rows). Used to
    validate the Hutchinson estimator.
    """
    W_diag = np.asarray(W_diag, dtype=float)
    n_d = W_diag.size
    f = sim.fields(m)
    acc = None
    for i in range(n_d):
        e = np.zeros(n_d)
        e[i] = W_diag[i]                    # row i of W J
        try:
            row = sim.Jtvec(m, e, f=f)
        except TypeError:
            row = sim.Jtvec(m, e)
        row = np.asarray(row, dtype=float)
        acc = row**2 if acc is None else acc + row**2
        if verbose and (i + 1) % 5 == 0:
            print(f"  exact row {i+1}/{n_d}", flush=True)
    return acc


def sensitivity_weights(jtj_diag, cell_volumes, clip=3e-3, exponent=1.0):
    """Cell weights from a JtJ diagonal, matching SimPEG's convention.

    Mirrors `UpdateSensitivityWeights.update` with
    `threshold_method="amplitude"` and `normalization_method="maximum"`:

        wr = sqrt(jtj_diag / vol**2) ** exponent
        wr = clip(wr, a_min=clip * wr.max())
        wr /= wr.max()

    `exponent` softens the weighting. SimPEG hardcodes the square root of
    jtj/vol**2; `exponent=0.5` makes that a fourth root overall, a common
    practical adjustment because the raw sqrt is aggressive. Measured here, it
    cuts the surface-to-depth contrast from 43x to 6.6x -- i.e. it removes most
    of the incentive to place structure at depth while keeping enough gradient
    to suppress the shallow "donut" the weighting exists to fix.

    `clip` defaults to 3e-3 rather than SimPEG's 1e-12. A Hutchinson estimate is
    stochastic, and an effectively-zero floor leaves far-field cells both
    unconstrained by the data and unregularized, which makes the Gauss-Newton
    system singular in those directions. 3e-3 was chosen from the measured
    50-sounding distribution: it floors ~8% of cells (all outside the survey
    footprint) while flooring *nothing* inside the core region, so the full
    surface-to-depth dynamic range is preserved where the data actually sense.
    """
    jtj_diag = np.asarray(jtj_diag, dtype=float)
    vol = np.asarray(cell_volumes, dtype=float)
    if jtj_diag.shape != vol.shape:
        raise ValueError(f"jtj_diag {jtj_diag.shape} != cell_volumes {vol.shape}")

    wr = np.sqrt(np.maximum(jtj_diag, 0.0) / vol**2)
    wr /= wr.max()                       # normalize before the power
    if exponent != 1.0:
        wr = wr**exponent
    if not np.all(np.isfinite(wr)):
        raise ValueError("non-finite sensitivity weights")
    wr = np.clip(wr, a_min=clip * wr.max(), a_max=np.inf)
    wr /= wr.max()
    return wr
