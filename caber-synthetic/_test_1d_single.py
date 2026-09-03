"""Single 1D inversion test, fixed beta (following the fixed-beta workflow of
2025-heagy-etal-tle/.../synthetic-data-1D-inversions-fixed-beta.ipynb).

Sounding at a survey corner OVER the overburden but FAR from the target
(x=340, y=200): the earth there is ~1D (conductive overburden over resistive
halfspace), so a 1D inversion should recover it well. Fixed beta (no cooling,
no target-misfit early stop) -> identical regularization for every sounding,
which is what makes the stitched section smooth.

Sweeps a few fixed beta values, reports phi_d for each, and plots the chosen
one: recovered conductivity profile vs the true layered model, plus the data
fit (observed vs predicted) for that sounding.
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np
import matplotlib
if __name__ == "__main__":
    matplotlib.use("Agg")   # headless only when run as a script; importing
                            # this module elsewhere must not hijack the backend
import matplotlib.pyplot as plt

import discretize
from simpeg import (
    maps, Data, data_misfit, inverse_problem,
    regularization, optimization, directives, inversion,
)
from simpeg.electromagnetics import time_domain as tdem

from _test_parametric import (
    rx_locs, rx_times, rx_x, rx_y,
    sigma_back, sigma_overburden, overburden_x, overburden_max_thickness,
)

# ---- 1D mesh (deeper core than the mill example: target reaches -300) ----
CS = 25.0
CORE = 400.0
NPAD = 12
PF = 1.3
THICK = discretize.utils.unpack_widths(
    [(CS, int(np.ceil(CORE / CS))), (CS, NPAD, PF)]
)
N_LAYERS = len(THICK) + 1
MESH1D = discretize.TensorMesh([np.r_[THICK, THICK[-1]]], origin="0")
DEPTH_TOP = np.r_[0.0, -np.cumsum(THICK)]  # top of each layer (z, downward neg)

REL_ERR = 0.10  # loosened from 0.05: the 3D-mesh data carries a ~20% (late)
# to ~2x (earliest channel) forward discrepancy vs the exact 1D forward, so a
# tighter fit forces the 1D to invent structure. 10% absorbs the late-time
# part. NOTE: the inversion target = chifact * n_data uses n_data, not the
# error level, so the misfit "target" number is unchanged; what changes is how
# much residual each channel is allowed before it drives structure.
NOISE_FLOOR = 1e-12


def overburden_thickness(x):
    x_c = overburden_x.mean()
    half_w = np.diff(overburden_x)[0] / 2
    if x < overburden_x[0] or x >= overburden_x[1]:
        return 0.0
    return overburden_max_thickness * (1 - ((x - x_c) / half_w) ** 2)


def make_inversion(loc, dobs, beta, maxIter=20, alpha_s=1.0 / CS, alpha_x=1.0):
    rx = tdem.receivers.PointMagneticFluxTimeDerivative(loc, rx_times, orientation="z")
    src = tdem.sources.CircularLoop(
        receiver_list=[rx], location=loc, orientation="z", radius=10,
        waveform=tdem.sources.StepOffWaveform(),
    )
    survey = tdem.Survey([src])
    sim = tdem.Simulation1DLayered(
        survey=survey, thicknesses=THICK, sigmaMap=maps.ExpMap(MESH1D),
    )
    data = Data(survey, dobs=dobs,
                standard_deviation=REL_ERR * np.abs(dobs) + NOISE_FLOOR)
    dmis = data_misfit.L2DataMisfit(simulation=sim, data=data)
    # reference = true background (sigma_back = 1e-3); insensitive (deep)
    # layers then park at the true background instead of an arbitrary start.
    reg = regularization.WeightedLeastSquares(
        MESH1D, alpha_s=alpha_s, alpha_x=alpha_x,
        reference_model=np.log(sigma_back) * np.ones(N_LAYERS),
    )
    opt = optimization.InexactGaussNewton(maxIter=maxIter)
    # FIXED beta: beta set on the inverse problem, NO BetaSchedule, NO
    # TargetMisfit -> beta is constant across all iterations (and, used the
    # same for every sounding, across all soundings).
    inv_prob = inverse_problem.BaseInvProblem(dmis, reg, opt, beta=beta)
    directives_list = [
        directives.UpdateSensitivityWeights(),
        directives.UpdatePreconditioner(),
        # stop at target misfit (beta stays fixed -- no cooling). chifact=1.1
        # so it stops as soon as phi_d crosses ~1.1*nD, catching the descent
        # near target instead of overshooting on the next GN step.
        directives.TargetMisfit(chifact=1.1),
    ]
    inv = inversion.BaseInversion(inv_prob, directives_list)
    return inv, sim, dmis


def main():
    import sys
    corner = sys.argv[1] if len(sys.argv) > 1 else "ovb"
    # "ovb": east corner over the overburden; "bg": west corner over clean
    # halfspace (overburden only starts at x=-100, so x=-380 is background).
    ix = int(np.argmin(rx_x)) if corner == "bg" else int(np.argmax(rx_x))
    iy = int(np.argmax(np.abs(rx_y)))
    s_ind = iy * len(rx_x) + ix
    loc = rx_locs[s_ind]
    t_ovb = overburden_thickness(loc[0])
    print(f"[{corner}] corner sounding: index {s_ind}, loc {loc}, "
          f"overburden thickness here = {t_ovb:.0f} m", flush=True)

    dobs = np.load("_dobs_cache.npy").reshape(len(rx_y), len(rx_x), len(rx_times))[iy, ix, :]
    print(f"dobs range: [{np.abs(dobs).min():.2e}, {np.abs(dobs).max():.2e}] T/s", flush=True)

    m0 = np.log(sigma_back) * np.ones(N_LAYERS)  # start at the true background
    n_d = len(rx_times)

    # fixed beta = 10, stop at target misfit (TargetMisfit in make_inversion)
    b_ch = 10.0
    inv, sim, dmis = make_inversion(loc, dobs, beta=b_ch)
    mopt = inv.run(m0.copy())
    phi_d = dmis(mopt)  # chi^2 = sum((r/std)^2); target = n_data
    print(f"\nfixed beta = {b_ch}, stopped at phi_d = {phi_d:.1f} (target {n_d})", flush=True)
    np.save("_1d_single_model.npy", mopt)

    # predicted data at chosen model
    dpred = sim.dpred(mopt)

    # ---- plot ----
    fig, ax = plt.subplots(1, 2, figsize=(11, 5))

    # (1) recovered conductivity profile vs true layered model
    z_plot = np.repeat(DEPTH_TOP, 2)[1:-1]
    sig_plot = np.repeat(np.exp(mopt)[:len(DEPTH_TOP) - 1], 2)
    ax[0].plot(sig_plot, z_plot, "C0-", lw=2, label="recovered 1D")
    # true: overburden over halfspace
    true_sig = np.full(len(DEPTH_TOP) - 1, sigma_back)
    layer_top = DEPTH_TOP[:-1]
    true_sig[layer_top > -t_ovb] = sigma_overburden
    ax[0].plot(np.repeat(true_sig, 2), z_plot, "k--", lw=1.5, label="true (ovb+halfspace)")
    ax[0].set_xscale("log")
    ax[0].set_xlabel("σ (S/m)")
    ax[0].set_ylabel("z (m)")
    ax[0].set_ylim([-400, 0])
    ax[0].set_xlim([5e-4, 2e-2])
    ax[0].legend()
    ax[0].grid(True, alpha=0.3, which="both")
    ax[0].set_title(f"corner sounding ({loc[0]:.0f},{loc[1]:.0f}): recovered vs true\n"
                    f"fixed β={b_ch:g}, φ_d={phi_d:.1f}/{n_d}")

    # (2) data fit
    ax[1].loglog(rx_times, -dobs, "ko", ms=5, label="observed")
    ax[1].loglog(rx_times, -dpred, "b-", lw=1.5, label="predicted (1D inv)")
    ax[1].set_xlabel("time (s)")
    ax[1].set_ylabel("-dB/dt (T/s)")
    ax[1].legend()
    ax[1].grid(True, alpha=0.3, which="both")
    ax[1].set_title("data fit at the corner sounding")

    plt.tight_layout()
    out = f"_plot_1d_single_{corner}.png"
    plt.savefig(out, dpi=110)
    print(f"saved -> {out}", flush=True)


if __name__ == "__main__":
    main()
