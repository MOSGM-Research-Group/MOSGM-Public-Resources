import numpy as np
import matplotlib.pyplot as plt

from analysis.sensitivity_scan import run_gated_sweep


if __name__ == "__main__":
    eps_vals, degenerate_frac = run_gated_sweep()

    eps_vals = np.asarray(eps_vals)
    degenerate_frac = np.asarray(degenerate_frac)

    plt.figure(figsize=(7, 4.5))
    plt.plot(eps_vals, degenerate_frac * 100, marker="o")

    plt.axhline(95, linestyle="--", linewidth=1)
    plt.text(
        eps_vals.min(),
        96,
        "MOND-degenerate region (≤5% deviation)",
        fontsize=9,
        verticalalignment="bottom",
    )

    plt.xlabel(r"Epsilon ($\epsilon$)")
    plt.ylabel("Degeneracy (%)")
    plt.title("MOSGM–MOND Degeneracy vs Gated Response ($r_0 = 8$ kpc)")

    plt.grid(True)
    plt.tight_layout()
    plt.show()
