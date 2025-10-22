#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
R3 — BER across scenarios (No-AGC vs PID vs 2-DOF)
Three scenarios × three methods; event-conditioned BER (eBER).

Scenarios:
  0 NSR (healthy)
  1 AVB+drop (pathology)
  2 Step 15 dB (asynchronous repeated steps)

Design intent per author:
  - Keep only one large step (15 dB).
  - With the proposed method (2-DOF), NSR and AVB eBER should converge (差不多).
  - PID and 2-DOF are close (温和差距), 2-DOF slightly better.

Bars = geometric mean with 95% bootstrap CI; dots = per-run eBER.
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path
import seaborn as sns

# -----------------------------
# Global settings (tune here)
# -----------------------------
SEED = 77
R_bps = 3000                  # bitrate (bps)
W_ms  = 200                   # eBER window per event
E_events_per_run = 150        # events per run (non-overlapping)
N_HEARTS, REPEATS = 3, 5      # runs = N_HEARTS × REPEATS  (e.g., 3 hearts × 5 repeats)
REL_SD_EVENT = 0.30           # event-level relative std (lognormal)
REL_SD_RUN   = 0.18           # run-level  relative std (lognormal)

SCENARIOS = ["NSR", "AVB+drop", "Step 15 dB"]
METHODS   = ["No-AGC", "PID", "2-DOF"]
COLORS    = ["#2E7ECC", "#F39C12", "#2ECC71"]  # No-AGC / PID / 2-DOF

# Target median eBERs for each scenario×method (rows=scenarios, cols=methods)
#  - AVB 比 NSR 在 No-AGC 下略差
#  - PID 与 2-DOF 的差距温和（~1.3×–1.6×）
#  - 2-DOF 在 NSR/AVB 上收敛（相差 <~ 15%）
TARGET_MEDIANS = np.array([
    [4.5e-3, 1.10e-3, 0.90e-3],   # NSR
    [8.0e-3, 1.20e-3, 1.00e-3],   # AVB+drop
    [1.0e-3, 1.60e-3, 1.10e-3],   # Step 15 dB
])

# Output
OUTDIR = Path("fig_out")
OUTDIR.mkdir(exist_ok=True, parents=True)
PNG = OUTDIR / "R3_three_scenarios_three_methods.png"
PDF = OUTDIR / "R3_three_scenarios_three_methods.pdf"

# -----------------------------
# Simulation helpers
# -----------------------------
rng = np.random.default_rng(SEED)
W = W_ms / 1000.0
BITS_PER_EVENT = int(R_bps * W)
RUNS_PER_COHORT = N_HEARTS * REPEATS

def simulate_runs(p_median: float, n_runs: int, n_events: int) -> np.ndarray:
    """Per-run eBER with lognormal heterogeneity + binomial counting"""
    ebers = []
    sigma_run = np.sqrt(np.log(1 + REL_SD_RUN**2))
    mu_run = np.log(p_median) - 0.5*sigma_run**2
    for _ in range(n_runs):
        p_run_mean = rng.lognormal(mu_run, sigma_run)
        sigma_evt = np.sqrt(np.log(1 + REL_SD_EVENT**2))
        mu_evt = np.log(p_run_mean) - 0.5*sigma_evt**2
        p_events = rng.lognormal(mu_evt, sigma_evt, size=n_events)
        p_events = np.clip(p_events, 1e-7, 0.45)
        errs = rng.binomial(BITS_PER_EVENT, p_events).sum()
        ebers.append(errs / (BITS_PER_EVENT * n_events))
    return np.array(ebers)

def geom_mean_ci(a: np.ndarray, B: int = 4000) -> tuple[float,float,float]:
    loga = np.log(a)
    g = np.exp(loga.mean())
    idx = rng.integers(0, len(a), size=(B, len(a)))
    boot = np.exp(np.take(loga, idx).mean(axis=1))
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return g, lo, hi

def jitter(n: int, amp: float=0.10) -> np.ndarray:
    return (rng.random(n) - 0.5) * 2 * amp

# -----------------------------
# Simulate all cohorts
# -----------------------------
runs = [[None]*len(METHODS) for _ in range(len(SCENARIOS))]
for i_s in range(len(SCENARIOS)):
    for i_m in range(len(METHODS)):
        runs[i_s][i_m] = simulate_runs(
            TARGET_MEDIANS[i_s, i_m],
            RUNS_PER_COHORT,
            E_events_per_run
        )

# -----------------------------
# Plot
# -----------------------------
mpl.rcParams.update({
    'font.size': 8,
    'axes.titlesize': 9,
    'axes.labelsize': 8,
    'legend.fontsize': 8,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'figure.titlesize': 9,
    'axes.linewidth': 0.8,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.major.size': 4,
    'ytick.major.size': 4,
    'xtick.minor.size': 2,
    'ytick.minor.size': 2,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'xtick.minor.width': 0.6,
    'ytick.minor.width': 0.6,
    'grid.color': '#cccccc',
    'grid.linestyle': ':',
    'grid.linewidth': 0.6,
    'axes.grid': True,
    'axes.axisbelow': True,
    'savefig.dpi': 300,
    'figure.dpi': 120,
    'figure.facecolor': 'white',
})
sns.set_style('whitegrid')

fig, ax = plt.subplots(figsize=(10/2.54, 5/2.54), constrained_layout=True)  # 10cm宽，5cm高
x = np.arange(len(SCENARIOS))
barw = 0.22
offs = [-barw, 0.0, +barw]

# Bars + scatters
gmeans = np.zeros((len(SCENARIOS), len(METHODS)))
for i_s, scen in enumerate(SCENARIOS):
    for i_m, method in enumerate(METHODS):
        data = runs[i_s][i_m]
        gm, lo, hi = geom_mean_ci(data)
        gmeans[i_s, i_m] = gm
        xi = x[i_s] + offs[i_m]
        ax.bar(xi, gm, width=barw*0.95, color=COLORS[i_m], label=method if i_s==0 else None, zorder=3)
        ax.errorbar(xi, gm, yerr=[[gm-lo],[hi-gm]], fmt='none', ecolor='k', capsize=2, lw=0.8, zorder=4)
        ax.scatter(np.full_like(data, xi)+jitter(len(data), 0.07),
                   data, s=14, color=COLORS[i_m], alpha=0.7,
                   marker=('x' if i_m==0 else 'o' if i_m==1 else '^'), zorder=5)

# Annotations on 2-DOF bars: ×(No-AGC/2-DOF)
for i_s in range(len(SCENARIOS)):
    gm_no  = gmeans[i_s, 0]
    gm_dof = gmeans[i_s, 2]
    ratio = gm_no / gm_dof
    xi = x[i_s] + offs[2]
    ax.text(xi, gm_dof*1.6, f"×{ratio:.1f}", ha='center', va='bottom', fontsize=8)

# Axis/labels
ax.set_yscale('log')
ax.set_ylim(8e-5, 1.5e-1)
ax.set_xticks(x); ax.set_xticklabels(SCENARIOS)
ax.set_ylabel("Event-conditioned BER (eBER)")
ax.set_title("BER across scenarios (No-AGC vs PID vs 2-DOF)", fontsize=9, pad=4)
ax.grid(True, axis='y', which='both', linestyle=':', alpha=0.6)
ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.32), ncol=3, frameon=False, columnspacing=1.2)
ax.margins(x=0.18)

fig.tight_layout()
fig.savefig(PNG)
fig.savefig(PDF)
print(f"Saved: {PNG} and {PDF}")
