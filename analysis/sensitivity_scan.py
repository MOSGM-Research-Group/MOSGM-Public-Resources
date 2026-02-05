import numpy as np


def run_gated_sweep(
    r_min=0.5,
    r_max=30.0,
    r_points=300,
    mb=1e10,
    a0=1.2e-10,
    r_gate=8.0,
    eps_min=-5,
    eps_max=-1,
    eps_points=20,
    degeneracy_threshold=0.05,
):
    """Run a gated MOSGM sweep and return epsilon and degeneracy fraction arrays."""
    r = np.linspace(r_min, r_max, r_points)
    gN = 6.67e-11 * mb * 1.989e30 / (r * 3.086e19) ** 2

    x = gN / a0
    gMOND = gN / (1 + 1 / x)

    eps_vals = np.logspace(eps_min, eps_max, eps_points)
    degenerate_frac = np.empty_like(eps_vals)

    dln_gbar = np.gradient(np.log(gN), np.log(r))
    gate = np.exp(-((r / r_gate) ** 2))

    for idx, eps in enumerate(eps_vals):
        agrad = eps * dln_gbar * gN * gate
        gMOSGM = gMOND + agrad

        rel_diff = np.abs(gMOSGM - gMOND) / gMOND
        degenerate_frac[idx] = np.mean(rel_diff < degeneracy_threshold)

    return eps_vals, degenerate_frac
