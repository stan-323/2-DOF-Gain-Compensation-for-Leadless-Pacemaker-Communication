#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Single-step (ΔG≈15 dB) — Event-conditioned BER (eBER) simulation/plot
Cohorts:
  1) Healthy (No-AGC)
  2) Pathological (No-AGC)
  3) PID (post-compensation)      <-- Healthy & Pathology shown together (overlaid markers)
  4) 2-DOF (post-compensation)    <-- Healthy & Pathology shown together (overlaid markers)

- Uses asynchronous repeated steps (step-train) to avoid "single-shot" BER artifacts
- eBER is counted in a fixed window W after each step
- Heterogeneity modeled with lognormal variability (run-level + event-level) and binomial counting
- Bars are geometric means with 95% bootstrap CI, dots are per-run eBER values
- Designed as an *experimental benchmark*; replace target medians with measured values when available

Author: (your name)
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ============================
# Experiment / model settings
# ============================
SEED = 20250924            # RNG seed for reproducibility
R_bps = 3000               # bitrate (bps) used for BER runs
W_ms  = 200                # eBER window length (ms) after each step event
E_events_per_run = 150     # number of step events per run (non-overlapping windows)
N_HEARTS, REPEATS = 3, 5   # runs per cohort = N_HEARTS * REPEATS

# Heterogeneity (tune to match your dispersion): relative std-deviation
REL_SD_EVENT = 0.30        # event-to-event variability
REL_SD_RUN   = 0.18        # run-to-run variability

# Target median eBERs (CHANGE THESE to your measured medians if you have them)
# No-AGC baselines (Healthy vs Pathology)
NSR_noAGC_median   = 2.4e-2
Path_noAGC_median  = 3.2e-2

# PID and 2-DOF (post-compensation). Keep them "close" per your expectation.
PID_healthy_median = 1.3e-3
PID_path_median    = 1.4e-3
DOF_healthy_median = 0.9e-3
DOF_path_median    = 1.0e-3

# Output paths
OUTDIR = Path("fig_out")
OUTDIR.mkdir(exist_ok=True, parents=True)
PNG = OUTDIR / "R3_single_step_four_cohorts_PID_close.png"
PDF = OUTDIR / "R3_single_step_four_cohorts_PID_close.pdf"

# ============================
# Simulation helpers
# ============================
rng = np.random.default_rng(SEED)
W = W_ms / 1000.0
BITS_PER_EVENT = int(R_bps * W)
RUNS_PER_COHORT = N_HEARTS * REPEATS

def simulate_runs(p_median: float, n_runs: int, n_events: int) -> np.ndarray:
    """Simulate per-run eBER with lognormal heterogeneity and binomial counting.
    Returns array of shape (n_runs,)."""
    ebers = []
    # run-level variability (lognormal around p_median)
    sigma_run = np.sqrt(np.log(1 + REL_SD_RUN**2))
    mu_run = np.log(p_median) - 0.5*sigma_run**2
    for _ in range(n_runs):
        p_run_mean = rng.lognormal(mu_run, sigma_run)
        # event-level variability around run mean
        sigma_evt = np.sqrt(np.log(1 + REL_SD_EVENT**2))
        mu_evt = np.log(p_run_mean) - 0.5*sigma_evt**2
        p_events = rng.lognormal(mu_evt, sigma_evt, size=n_events)
        p_events = np.clip(p_events, 1e-7, 0.45)
        errs = rng.binomial(BITS_PER_EVENT, p_events).sum()
        ebers.append(errs / (BITS_PER_EVENT * n_events))
    return np.array(ebers)

def geom_mean_and_ci(a: np.ndarray, B: int = 5000) -> tuple[float,float,float]:
    """Geometric mean + bootstrap 95% CI (on log scale)."""
    loga = np.log(a)
    g = np.exp(loga.mean())
    idx = rng.integers(0, len(a), size=(B, len(a)))
    boot = np.exp(np.take(loga, idx).mean(axis=1))
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return g, lo, hi

def jitter(n: int, amp: float = 0.08) -> np.ndarray:
    return (rng.random(n) - 0.5) * 2 * amp

# ============================
# Run simulation
# ============================
nsr_no   = simulate_runs(NSR_noAGC_median,  RUNS_PER_COHORT, E_events_per_run)
path_no  = simulate_runs(Path_noAGC_median, RUNS_PER_COHORT, E_events_per_run)

pid_h    = simulate_runs(PID_healthy_median, RUNS_PER_COHORT, E_events_per_run)
pid_p    = simulate_runs(PID_path_median,    RUNS_PER_COHORT, E_events_per_run)

dof_h    = simulate_runs(DOF_healthy_median, RUNS_PER_COHORT, E_events_per_run)
dof_p    = simulate_runs(DOF_path_median,    RUNS_PER_COHORT, E_events_per_run)

# ============================
# Plot
# ============================
labels = ["Healthy\n(No-AGC)",
          "Pathological\n(No-AGC)",
          "PID\n(post-comp)",
          "2-DOF\n(post-comp)"]

fig, ax = plt.subplots(figsize=(11.5, 4.8))
x = np.arange(4); w = 0.62

def add_bar(i: int, data: np.ndarray, color: str) -> float:
    gm, lo, hi = geom_mean_and_ci(data)
    ax.bar(x[i], gm, width=w, color=color)
    ax.errorbar(x[i], gm, yerr=[[gm-lo],[hi-gm]], fmt='none', ecolor='k', capsize=3, lw=1.0)
    return gm

# Colors
c_no_h, c_no_p, c_pid, c_dof = "#2E7ECC", "#C0392B", "#F39C12", "#2ECC71"

# Bars
b1 = add_bar(0, nsr_no, c_no_h)
b2 = add_bar(1, path_no, c_no_p)
pid_combined = np.hstack([pid_h, pid_p])
dof_combined = np.hstack([dof_h, dof_p])
b3 = add_bar(2, pid_combined, c_pid)
b4 = add_bar(3, dof_combined, c_dof)

# Scatter overlays
ax.scatter(np.full_like(nsr_no, x[0]) + jitter(len(nsr_no)), nsr_no, s=18, color=c_no_h, marker='x', alpha=0.85, label="Healthy runs")
ax.scatter(np.full_like(path_no, x[1]) + jitter(len(path_no)), path_no, s=18, color=c_no_p, marker='o', alpha=0.85, label="Pathology runs")

ax.scatter(np.full_like(pid_h, x[2])+jitter(len(pid_h),0.05), pid_h, s=22, color="#1B9E77", marker='^', alpha=0.95, label="PID—Healthy")
ax.scatter(np.full_like(pid_p, x[2])+jitter(len(pid_p),0.05), pid_p, s=22, color="#D95F02", marker='s', alpha=0.95, label="PID—Pathology")

ax.scatter(np.full_like(dof_h, x[3])+jitter(len(dof_h),0.05), dof_h, s=22, color="#1B9E77", marker='^', alpha=0.95, label="2-DOF—Healthy")
ax.scatter(np.full_like(dof_p, x[3])+jitter(len(dof_p),0.05), dof_p, s=22, color="#D95F02", marker='s', alpha=0.95, label="2-DOF—Pathology")

# Annotations
ratio_pid_dof = b3 / b4
imp_h = (1 - b4 / b1) * 100
imp_p = (1 - b4 / b2) * 100
ax.text(x[3], b4*1.55, f"PID/2-DOF ×{ratio_pid_dof:.1f}", ha='center', va='bottom', fontsize=9)
ax.text(x[0], b4*0.72, f"{imp_h:.0f}% vs Healthy No-AGC", ha='center', va='top', fontsize=9, color=c_dof)
ax.text(x[1], b4*0.60, f"{imp_p:.0f}% vs Pathology No-AGC", ha='center', va='top', fontsize=9, color=c_dof)

# Axes
ax.set_yscale('log')
ax.set_ylim(6e-5, 1.5e-1)
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_ylabel("Event-conditioned BER (eBER)")
ax.set_title("Single-step (ΔG≈15 dB) — No-AGC vs PID vs 2-DOF\nPID close to 2-DOF; both converge across Healthy/Pathological")
ax.grid(True, axis='y', which='both', linestyle=':', alpha=0.6)
ax.legend(loc='upper right', ncol=2, fontsize=8, framealpha=0.9)

# Footer
ax.text(0.01, -0.22,
        f"Protocol: asynchronous repeated steps (ΔG≈15 dB), window W={W_ms} ms, bitrate {R_bps} bps, "
        f"events/run ≈{E_events_per_run}, runs={RUNS_PER_COHORT} per cohort. "
        "Bars: geometric mean with 95% bootstrap CI; dots: per-run eBER.",
        transform=ax.transAxes, fontsize=9)

fig.tight_layout()
fig.savefig(PNG, dpi=300, bbox_inches='tight')
fig.savefig(PDF, bbox_inches='tight')

print(f"Saved: {PNG} and {PDF}")
