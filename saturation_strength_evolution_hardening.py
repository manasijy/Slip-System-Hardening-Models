#!/usr/bin/env python3
"""
Saturation strength evolution model: g(Γ) visualization.

Model (FEPX saturation-evolution hardening):
  g_s(γdot) = gs0 * (γdot/γdot_s0)**m_prime
  gdot = h0 * ((g_s(γdot) - g)/(g_s(γdot) - g0))**n * γdot
  ⇒ Using Γ = ∫γdot dt:  dg/dΓ = h0 * ((g_s(γdot) - g)/(g_s(γdot) - g0))**n

Use modes:
  - interactive_constant: sliders for gs0, g0, h0, n, m_prime, and rate ratio r = γdot/γdot_s0
    Analytic solution (Voce-form) with g_s_eff = gs0 * r**m_prime.
  - sweep: lists for parameters (including r) to compare individual/combined effects (analytic).
  - variable_rate_mode: prescribe r(Γ) and integrate numerically in Γ with RK4.

Notes:
  - If m_prime = 0, reduces to base model with constant g_s = gs0.
  - Require gs_eff > g0 to avoid degeneracy; code guards and annotates when not satisfied.
  - Units: pick any consistent force/area and 1/s; Γ is dimensionless accumulated slip.
"""

import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# --------------------
# Configuration
# --------------------
interactive_constant = True   # Analytic, single r with sliders
sweep_mode = False            # Analytic, many combinations
variable_rate_mode = False    # Numerical RK4 with r(Γ) profile

# Γ range
Gamma_max_default = 5.0
Gamma_points = 1401

# Defaults for interactive mode
params_init = dict(gs0=150.0, g0=80.0, h0=200.0, n=1.0, m_prime=0.5, r=1.0)

# Grids for sweep mode (edit to taste)
sweep_params = dict(
    gs0=[140.0, 160.0],
    g0=[70.0, 90.0],
    h0=[100.0, 200.0],
    n=[0.8, 1.0, 2.0],
    m_prime=[0.0, 0.5, 1.0],
    r=[0.5, 1.0, 2.0, 5.0],   # r = γdot / γdot_s0
)

# Variable-rate profile (used only if variable_rate_mode=True)
var_rate = dict(
    profile="step",   # "constant", "step", "ramp", "sin"
    r0=1.0,           # base ratio for constant/sin, start for ramp
    r1=0.5,           # before step
    r2=3.0,           # after step
    Gamma_c=1.0,      # step location
    k=0.5,            # ramp slope (r = r0 + k*Γ, clipped ≥ r_min)
    a=0.5,            # sinusoid amplitude (r = r0*(1 + a*sin(...)))
    T=2.0,            # sinusoid period in Γ
    r_min=1e-6,       # floor to keep r positive
)

# --------------------
# Core model utilities
# --------------------
def gs_eff_from_r(gs0, r, m_prime):
    r = np.asarray(r, dtype=float)
    return gs0 * np.power(np.clip(r, 0.0, None), m_prime)

def g_voce_analytic(Gamma, gs_eff, g0, h0, n):
    """Analytic solution for constant g_s = gs_eff (Voce-type)."""
    Gamma = np.asarray(Gamma, dtype=float)
    gs_eff, g0, h0, n = float(gs_eff), float(g0), float(h0), float(n)

    # Degenerate/no-hardening cases
    if h0 == 0.0 or np.isclose(gs_eff, g0):
        return np.full_like(Gamma, g0, dtype=float)

    gap = gs_eff - g0

    if np.isclose(n, 1.0):
        return gs_eff - gap * np.exp(-(h0 / gap) * Gamma)

    factor = 1.0 + (n - 1.0) * h0 * Gamma / gap
    if n < 1.0:
        factor = np.clip(factor, 0.0, None)
    with np.errstate(all='ignore'):
        delta = gap * np.power(factor, 1.0 / (1.0 - n))
    g = gs_eff - delta
    return np.where(np.isfinite(g), g, np.nan)

def r_of_Gamma(Gamma, cfg):
    prof = cfg.get("profile", "constant")
    r_min = cfg.get("r_min", 1e-6)
    if prof == "constant":
        r = np.full_like(Gamma, cfg.get("r0", 1.0), dtype=float)
    elif prof == "step":
        r = np.where(Gamma < cfg.get("Gamma_c", 1.0), cfg.get("r1", 0.5), cfg.get("r2", 3.0))
    elif prof == "ramp":
        r = cfg.get("r0", 1.0) + cfg.get("k", 0.5) * Gamma
    elif prof == "sin":
        r = cfg.get("r0", 1.0) * (1.0 + cfg.get("a", 0.5) * np.sin(2.0 * np.pi * Gamma / cfg.get("T", 2.0)))
    else:
        raise ValueError(f"Unknown rate profile: {prof}")
    return np.clip(r, r_min, None)

def integrate_variable_rate(Gamma, gs0, g0, h0, n, m_prime, rate_cfg):
    """RK4 integration of dg/dΓ with r(Γ) profile."""
    Gamma = np.asarray(Gamma, dtype=float)
    dG = np.diff(Gamma)
    if not np.allclose(np.max(np.abs(dG - dG[0])), 0.0):
        raise ValueError("Gamma must be uniform for RK4 implementation.")

    g = np.empty_like(Gamma)
    g[0] = g0
    dΓ = dG[0]
    r_vals = r_of_Gamma(Gamma, rate_cfg)

    def rhs(i, gi):
        # Evaluate at index i (Γ_i) using local r value
        r_i = r_vals[i]
        gs_i = gs_eff_from_r(gs0, r_i, m_prime)
        gap = gs_i - g0
        if h0 == 0.0 or gap <= 0.0:
            return 0.0
        return h0 * ((gs_i - gi) / gap) ** n

    for i in range(len(Gamma) - 1):
        gi = g[i]
        k1 = rhs(i, gi)
        # Midpoint estimates use nearest indices for r; acceptable for small dΓ
        k2 = rhs(i, gi + 0.5 * dΓ * k1)
        k3 = rhs(i, gi + 0.5 * dΓ * k2)
        k4 = rhs(i + 1, gi + dΓ * k3)  # use r at i+1
        g[i + 1] = gi + (dΓ / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        # Optional: clamp to [g0, gs_i] for stability if overshoot occurs
        gs_next = gs_eff_from_r(gs0, r_vals[i + 1], m_prime)
        g[i + 1] = np.clip(g[i + 1], min(g0, gs_next), max(g0, gs_next))

    return g, r_vals

def gamma_max_for_visibility(gs_eff, g0, h0, n, default=Gamma_max_default):
    if h0 <= 0 or np.isclose(gs_eff, g0):
        return default
    if n < 1.0:
        return max(0.2 * default, (gs_eff - g0) / ((1.0 - n) * h0))
    return default

# --------------------
# Plotting
# --------------------
def plot_interactive_constant():
    fig, ax = plt.subplots(figsize=(7.8, 5.4))
    plt.subplots_adjust(left=0.10, right=0.98, top=0.90, bottom=0.28)

    # Add equation text
    eq_text = (
        r"$\frac{dg}{d\Gamma} = h_0 \left(\frac{g_s - g}{g_s - g_0}\right)^{n}$"
        "\n"
        r"$g_s = g_{s0} \left(\frac{\dot{\gamma}}{\dot{\gamma}_{s0}}\right)^{m'}$"
    )
    fig.text(0.5, 0.5, eq_text, ha="center", va="center", fontsize=12)

    gs00, g00, h00, n0, mp0, r0 = (
        params_init["gs0"], params_init["g0"], params_init["h0"],
        params_init["n"], params_init["m_prime"], params_init["r"]
    )
    gs_eff0 = gs_eff_from_r(gs00, r0, mp0)
    Gmax = gamma_max_for_visibility(gs_eff0, g00, h00, n0)
    Gamma = np.linspace(0.0, Gmax, Gamma_points)
    g_r = g_voce_analytic(Gamma, gs_eff0, g00, h00, n0)
    g_ref = g_voce_analytic(Gamma, gs_eff_from_r(gs00, 1.0, mp0), g00, h00, n0)

    l_cur, = ax.plot(Gamma, g_r, lw=2.4, color='C0', label=f"r={r0:g}")
    l_ref, = ax.plot(Gamma, g_ref, lw=1.8, ls='--', color='C1', label="r=1 (reference)")
    ax.set_xlabel("Γ (accumulated slip)")
    ax.set_ylabel("g(Γ)")
    ax.set_title("Saturation-strength evolution (constant rate): analytic Voce-form")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")

    # Sliders
    axcolor = 'lightgoldenrodyellow'
    ax_gs0 = plt.axes([0.10, 0.21, 0.80, 0.03], facecolor=axcolor)
    ax_g0  = plt.axes([0.10, 0.17, 0.80, 0.03], facecolor=axcolor)
    ax_h0  = plt.axes([0.10, 0.13, 0.80, 0.03], facecolor=axcolor)
    ax_n   = plt.axes([0.10, 0.09, 0.80, 0.03], facecolor=axcolor)
    ax_mp  = plt.axes([0.10, 0.05, 0.80, 0.03], facecolor=axcolor)
    ax_r   = plt.axes([0.10, 0.01, 0.80, 0.03], facecolor=axcolor)

    s_gs0 = Slider(ax_gs0, 'gs0',  1.0,  500.0, valinit=gs00, valstep=0.5)
    s_g0  = Slider(ax_g0,  'g0',   0.0,  499.0, valinit=g00,  valstep=0.5)
    s_h0  = Slider(ax_h0,  'h0',   0.0,  1000.0, valinit=h00, valstep=1.0)
    s_n   = Slider(ax_n,   'n',    0.1,  5.0,    valinit=n0,  valstep=0.01)
    s_mp  = Slider(ax_mp,  "m'",   0.0,  2.0,    valinit=mp0, valstep=0.01)
    s_r   = Slider(ax_r,   'r',    0.01, 10.0,   valinit=r0,  valstep=0.01)

    def update(_):
        gs0, g0, h0, n, mp, r = s_gs0.val, s_g0.val, s_h0.val, s_n.val, s_mp.val, s_r.val
        # Maintain physical clarity: gs_eff >= g0
        gs_eff = gs_eff_from_r(gs0, r, mp)
        if gs_eff < g0:
            s_g0.set_val(gs_eff)  # snap g0 down to gs_eff
            g0 = gs_eff
        Gmax = gamma_max_for_visibility(gs_eff, g0, h0, n)
        Gamma = np.linspace(0.0, Gmax, Gamma_points)
        l_cur.set_data(Gamma, g_voce_analytic(Gamma, gs_eff, g0, h0, n))
        l_cur.set_label(f"r={r:g}")
        # reference r=1
        g_ref = g_voce_analytic(Gamma, gs_eff_from_r(gs0, 1.0, mp), g0, h0, n)
        l_ref.set_data(Gamma, g_ref)
        ax.relim(); ax.autoscale_view()
        ax.legend(loc="best")
        fig.canvas.draw_idle()

    for s in (s_gs0, s_g0, s_h0, s_n, s_mp, s_r):
        s.on_changed(update)

    plt.show()
# ...existing code...

def plot_sweep(sweep):
    fig, ax = plt.subplots(figsize=(7.8, 5.4))
    keys = ["gs0", "g0", "h0", "n", "m_prime", "r"]
    grids = [sweep[k] if isinstance(sweep[k], (list, tuple, np.ndarray)) else [sweep[k]] for k in keys]
    # Choose a Γ_max that works across combinations (based on n and gs_eff)
    Gcands = []
    for gs0, g0, h0, n, mp, r in itertools.product(*grids):
        gs_eff = gs_eff_from_r(gs0, r, mp)
        Gcands.append(gamma_max_for_visibility(gs_eff, g0, h0, n))
    Gmax = min(max(Gcands), Gamma_max_default) if Gcands else Gamma_max_default
    Gamma = np.linspace(0.0, Gmax, Gamma_points)

    for (gs0, g0, h0, n, mp, r) in itertools.product(*grids):
        gs_eff = gs_eff_from_r(gs0, r, mp)
        g = g_voce_analytic(Gamma, gs_eff, g0, h0, n)
        ax.plot(Gamma, g, lw=1.8,
                label=f"gs0={gs0:g}, g0={g0:g}, h0={h0:g}, n={n:g}, m'={mp:g}, r={r:g}")

    ax.set_xlabel("Γ (accumulated slip)")
    ax.set_ylabel("g(Γ)")
    ax.set_title("Saturation-strength evolution (constant rates): sweep of parameters")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=7, ncols=1)
    plt.tight_layout()
    plt.show()

def plot_variable_rate(gs0, g0, h0, n, m_prime, rate_cfg):
    fig, ax1 = plt.subplots(figsize=(7.8, 5.4))
    Gmax = Gamma_max_default
    Gamma = np.linspace(0.0, Gmax, Gamma_points)
    g_num, r_vals = integrate_variable_rate(Gamma, gs0, g0, h0, n, m_prime, rate_cfg)

    # Also show an "equivalent constant rate" for reference (r = mean r)
    r_eq = float(np.mean(r_vals))
    g_eq = g_voce_analytic(Gamma, gs_eff_from_r(gs0, r_eq, m_prime), g0, h0, n)

    ax1.plot(Gamma, g_num, lw=2.2, color='C0', label="variable-rate (numerical)")
    ax1.plot(Gamma, g_eq, lw=1.6, ls='--', color='C1', label=f"constant-rate eq. (r̄={r_eq:.2g})")
    ax1.set_xlabel("Γ (accumulated slip)")
    ax1.set_ylabel("g(Γ)")
    ax1.set_title(f"Variable rate profile: {rate_cfg.get('profile','constant')}")
    ax1.grid(True, alpha=0.25)
    ax1.legend(loc="best")

    # Secondary axis for r(Γ)
    ax2 = ax1.twinx()
    ax2.plot(Gamma, r_vals, lw=1.0, color='tab:gray', alpha=0.7, label="r(Γ)")
    ax2.set_ylabel("r = γdot / γdot_s0")
    ax2.set_ylim(bottom=0.0)
    plt.tight_layout()
    plt.show()

# --------------------
# Main
# --------------------
def main():
    if interactive_constant:
        plot_interactive_constant()
    if sweep_mode:
        plot_sweep(sweep_params)
    if variable_rate_mode:
        p = params_init
        plot_variable_rate(p["gs0"], p["g0"], p["h0"], p["n"], p["m_prime"], var_rate)

if __name__ == "__main__":
    main()
