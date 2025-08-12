#!/usr/bin/env python3
"""
Plot g(Γ) for the integral solution of the base Voce-type hardening model.

Model:
  dg/dΓ = h0 * ((gs - g)/(gs - g0))**n,  g(0) = g0

Solution:
  - n = 1: g = gs - (gs - g0)*exp( -h0*Γ/(gs - g0) )
  - n != 1: g = gs - (gs - g0) * [ 1 + (n - 1)*h0*Γ/(gs - g0) ]**(1/(1 - n))

Modes:
  - interactive = True: Matplotlib sliders for gs, g0, n, h0
  - interactive = False: define lists for parameters; plots all combinations (individual + combined effects)
"""

import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# --------------------
# Configuration
# --------------------
interactive = True  # Set False for sweep mode

# Γ range (accumulated slip). You can adjust these defaults.
Gamma_max_default = 5.0
Gamma_points = 1201

# Initial parameter values for interactive mode
params_init = dict(gs=150.0, g0=80.0, h0=200.0, n=1.0)

# Parameter grids for sweep mode (each can be a single value or a list)
# Examples below illustrate individual and combined sweeps; edit as needed.
sweep_params = dict(
    gs=[140.0, 160.0],
    g0=[70.0, 90.0],
    h0=[100.0, 200.0],
    n=[0.5, 1.0, 2.0],
)

# --------------------
# Model implementation
# --------------------
def g_of_Gamma(Gamma, gs, g0, h0, n):
    """
    Vectorized evaluation of g(Γ) with robust handling of edge cases.
    Gamma: array
    Returns: array with same shape
    """
    Gamma = np.asarray(Gamma, dtype=float)
    gs, g0, h0, n = float(gs), float(g0), float(h0), float(n)

    # No hardening or degenerate saturation gap
    if h0 == 0.0 or np.isclose(gs, g0):
        return np.full_like(Gamma, g0, dtype=float)

    gap = gs - g0
    if np.isclose(n, 1.0):
        # Voce exponential
        return gs - gap * np.exp(- (h0 / gap) * Gamma)

    # General case
    factor = 1.0 + (n - 1.0) * h0 * Gamma / gap

    # If n < 1, factor must stay >= 0; clip for numerical stability
    if n < 1.0:
        factor = np.clip(factor, 0.0, None)

    with np.errstate(all='ignore'):
        delta = gap * np.power(factor, 1.0 / (1.0 - n))

    g = gs - delta
    # Clean any tiny numerical issues
    g = np.where(np.isfinite(g), g, np.nan)
    return g

def gamma_max_for_visibility(gs, g0, h0, n, default=Gamma_max_default):
    """
    Choose a Γ_max that shows interesting parts of the curve:
      - If n<1, stop at the theoretical limit Γ_max = (gs - g0)/((1-n)*h0).
      - Else, use default.
    """
    gs, g0, h0, n = float(gs), float(g0), float(h0), float(n)
    if h0 <= 0 or np.isclose(gs, g0):
        return default
    if n < 1.0:
        return max(0.2 * default, (gs - g0) / ((1.0 - n) * h0))
    return default

# --------------------
# Plotting Utilities
# --------------------
def plot_single(ax, gs, g0, h0, n):
    ax.clear()
    Gmax = gamma_max_for_visibility(gs, g0, h0, n)
    Gamma = np.linspace(0.0, Gmax, Gamma_points)
    g = g_of_Gamma(Gamma, gs, g0, h0, n)

    ax.plot(Gamma, g, lw=2.2, color='C0',
            label=f"gs={gs:g}, g0={g0:g}, h0={h0:g}, n={n:g}")

    # Mark the finite Γ limit for n<1
    if n < 1.0 and h0 > 0 and gs > g0:
        G_lim = (gs - g0) / ((1.0 - n) * h0)
        ax.axvline(G_lim, color='k', ls='--', lw=1.0, alpha=0.6)
        ax.text(G_lim, ax.get_ylim()[0], " Γ_limit", rotation=90,
                va='bottom', ha='left', fontsize=9, alpha=0.7)

    ax.set_xlabel("Γ (accumulated slip)")
    ax.set_ylabel("g(Γ)")
    ax.set_title("Integral solution: g(Γ) for base Voce-type hardening")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")

def plot_sweep(ax, sweep):
    ax.clear()
    keys = ["gs", "g0", "h0", "n"]
    grids = [sweep[k] if isinstance(sweep[k], (list, tuple, np.ndarray)) else [sweep[k]] for k in keys]

    # Pick a Γ_max that works across combinations (favor visibility; handle n<1)
    Gmax_candidates = []
    for gs, g0, h0, n in itertools.product(*grids):
        Gmax_candidates.append(gamma_max_for_visibility(gs, g0, h0, n))
    Gmax = min(max(Gmax_candidates), Gamma_max_default) if Gmax_candidates else Gamma_max_default

    Gamma = np.linspace(0.0, Gmax, Gamma_points)

    for i, (gs, g0, h0, n) in enumerate(itertools.product(*grids)):
        g = g_of_Gamma(Gamma, gs, g0, h0, n)
        ax.plot(Gamma, g, lw=1.8, label=f"gs={gs:g}, g0={g0:g}, h0={h0:g}, n={n:g}")

    ax.set_xlabel("Γ (accumulated slip)")
    ax.set_ylabel("g(Γ)")
    ax.set_title("Sweep: individual and combined effects on g(Γ)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8, ncols=1)

# --------------------
# Main
# --------------------
def main():
    if interactive:
        fig, ax = plt.subplots(figsize=(7.6, 5.2))
        plt.subplots_adjust(left=0.10, right=0.98, top=0.92, bottom=0.32)  # More space at bottom

        # Add equation text
        eq_text = (
                   r"$\frac{dg}{d\gamma} = h_0 \left(\frac{g_s - g}{g_s - g_0}\right)^{n}$"
                   )
        fig.text(0.5, 0.5, eq_text, ha="center", va="center", fontsize=12)

        gs0, g00, h00, n0 = params_init["gs"], params_init["g0"], params_init["h0"], params_init["n"]
        plot_single(ax, gs0, g00, h00, n0)

        # Sliders
        axcolor = 'lightgoldenrodyellow'
        ax_gs = plt.axes([0.10, 0.18, 0.80, 0.03], facecolor=axcolor)
        ax_g0 = plt.axes([0.10, 0.14, 0.80, 0.03], facecolor=axcolor)
        ax_h0 = plt.axes([0.10, 0.10, 0.80, 0.03], facecolor=axcolor)
        ax_n  = plt.axes([0.10, 0.06, 0.80, 0.03], facecolor=axcolor)

        s_gs = Slider(ax_gs, 'gs',  1.0,  500.0, valinit=gs0, valstep=0.5)
        s_g0 = Slider(ax_g0, 'g0',  0.0,  499.0, valinit=g00, valstep=0.5)
        s_h0 = Slider(ax_h0, 'h0',  0.0,  1000.0, valinit=h00, valstep=1.0)
        s_n  = Slider(ax_n,  'n',   0.1,  5.0,    valinit=n0,  valstep=0.01)

        def update(_):
            gs, g0, h0, n = s_gs.val, s_g0.val, s_h0.val, s_n.val
            # Keep gs >= g0 for physical clarity
            if gs < g0:
                # Snap gs up to g0
                s_gs.set_val(g0)
                gs = g0
            plot_single(ax, gs, g0, h0, n)
            fig.canvas.draw_idle()

        for s in (s_gs, s_g0, s_h0, s_n):
            s.on_changed(update)

        plt.show()

    else:
        fig, ax = plt.subplots(figsize=(7.6, 5.2))
        plot_sweep(ax, sweep_params)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()
