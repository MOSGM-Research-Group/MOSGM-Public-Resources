import numpy as np


def run_gated_sweep():
    # Same toy setup
    r = np.linspace(0.5, 30.0, 300)
    Mb = 1e10  # Msun
    gN = 6.67e-11 * Mb * 1.989e30 / (r * 3.086e19) ** 2

    # MOND baseline
    a0 = 1.2e-10
    x = gN / a0
    gMOND = gN / (1 + 1 / x)  # μ(x)=x/(1+x)

    results = []

    # Epsilon values (logarithmic spacing)
    eps_vals = np.logspace(-5, -1, 20)

    for eps in eps_vals:
        # Gated MOSGM
        dln_gbar = np.gradient(np.log(gN), np.log(r))
        gate = np.exp(-(r / 8.0) ** 2)  # r_gate = 8 kpc
        agrad = eps * dln_gbar * gN * gate
        gMOSGM = gMOND + agrad

        # Degeneracy calculation
        rel_diff = np.abs(gMOSGM - gMOND) / gMOND
        degenerate_frac = np.mean(rel_diff < 0.05) * 100

        results.append((eps, degenerate_frac))
        print(
            f"μ = {eps:.1e} | r_gate = 8 kpc | degeneracy = {degenerate_frac:.1f}%"
        )

    return results
