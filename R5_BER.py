# plot_ber_ieee_no_improve_legend.py
# -*- coding: utf-8 -*-
"""
BER grouped-bar plot for IEEE single-column (10 cm width).
Modifications:
 - remove legend entry for improvement lines
 - improvement lines colored same as their bars
 - keep bars legend (compact, fontsize 8)
 - Times New Roman, axis font size 10, legend 8
 - bar edge linewidth = 1.0, axis spine linewidth = 1.0
 - y-axis (log) fixed 1e-5 .. 1e-1
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ===== User settings =====
CSV_PATH = Path("/mnt/data/3.csv")   # <- 改成你的 CSV 路径（或把文件放在 /mnt/data/3.csv）
OUT_PNG = Path("ber_plot_ieee_no_improve_legend.png")
OUT_PDF = Path("ber_plot_ieee_no_improve_legend.pdf")

WIDTH_CM = 10.0
HEIGHT_CM = 8.0
# =========================

def load_or_synth(csv_path):
    if csv_path.exists():
        df = pd.read_csv(csv_path)
    else:
        # synthetic preview (3 hearts x 6 rates)
        rates = np.array([500,1000,2000,3000,4000,5000])
        rows = []
        np.random.seed(0)
        for h in [1,2,3]:
            base_no = np.array([2.9e-4, 4.1e-4, 7.8e-4, 1.05e-3, 1.2e-3, 1.4e-3])
            base_no = base_no * np.exp(np.random.normal(0, 0.12, size=base_no.shape))
            improve_pid = np.array([0.55,0.60,0.67,0.73,0.76,0.84])
            ratio_2dof_vs_pid = np.array([0.80,0.75,0.70,0.65,0.60,0.55])
            pid = base_no * (1 - improve_pid)
            dof = pid * ratio_2dof_vs_pid
            for r, R in enumerate(rates):
                rows.append({"heart":h, "rate_bps":int(R),
                             "NoAGC":base_no[r], "PID":pid[r], "2DOF":dof[r]})
        df = pd.DataFrame(rows)
    return df

# load
df = load_or_synth(CSV_PATH)

# checks
required = {"heart","rate_bps","NoAGC","PID","2DOF"}
missing = required - set(df.columns)
if missing:
    raise SystemExit(f"ERROR: Missing columns in CSV: {missing}")

# compute improvements if absent
if "Improve_PID" not in df.columns:
    df["Improve_PID"] = 1.0 - df["PID"]/df["NoAGC"]
if "Improve_2DOF" not in df.columns:
    df["Improve_2DOF"] = 1.0 - df["2DOF"]/df["NoAGC"]

# numeric
for c in ["heart","rate_bps"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")
for c in ["NoAGC","PID","2DOF","Improve_PID","Improve_2DOF"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# aggregate
agg = (
    df.groupby("rate_bps", as_index=False)
      .agg(NoAGC_mean=("NoAGC","mean"), NoAGC_sd=("NoAGC","std"),
           PID_mean=("PID","mean"),     PID_sd=("PID","std"),
           DOF_mean=("2DOF","mean"),    DOF_sd=("2DOF","std"),
           ImpPID_mean=("Improve_PID","mean"),   ImpPID_sd=("Improve_PID","std"),
           ImpDOF_mean=("Improve_2DOF","mean"), ImpDOF_sd=("Improve_2DOF","std"))
)

rates = agg["rate_bps"].to_numpy()
x = np.arange(len(rates))
bar_w = 0.24

# figure size
cm2inch = 1.0/2.54
fig_w = WIDTH_CM * cm2inch
fig_h = HEIGHT_CM * cm2inch

# style
plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 10,
    "axes.linewidth": 1.0,
    "xtick.direction": "in",
    "ytick.direction": "in",
})

fig, ax = plt.subplots(figsize=(fig_w, fig_h))

# colors
colors = {"NoAGC":"#d9534f", "PID":"#4aa3c6", "2DOF":"#f29f3d"}

# bars (solid, black edge width=1)
ax.bar(x - bar_w, agg["NoAGC_mean"], bar_w, yerr=agg["NoAGC_sd"], capsize=2,
       color=colors["NoAGC"], edgecolor="black", linewidth=1.0, label="No AGC", zorder=2)
ax.bar(x,          agg["PID_mean"],   bar_w, yerr=agg["PID_sd"], capsize=2,
       color=colors["PID"], edgecolor="black", linewidth=1.0, label="PID", zorder=2)
ax.bar(x + bar_w,  agg["DOF_mean"],   bar_w, yerr=agg["DOF_sd"], capsize=2,
       color=colors["2DOF"], edgecolor="black", linewidth=1.0, label="2-DOF", zorder=2)

# per-heart scatter (match bar colors)
rng = np.random.default_rng(1234)
for i, R in enumerate(rates):
    sub = df[df["rate_bps"]==R]
    jit = (rng.random(len(sub)) - 0.5) * (bar_w * 0.6)
    ax.scatter(np.full(len(sub), x[i]-bar_w) + jit, sub["NoAGC"],
               s=16, color=colors["NoAGC"], edgecolor='0.2', alpha=0.6, zorder=3)
    ax.scatter(np.full(len(sub), x[i]) + jit, sub["PID"],
               s=16, color=colors["PID"], edgecolor='0.2', alpha=0.6, zorder=3)
    ax.scatter(np.full(len(sub), x[i]+bar_w) + jit, sub["2DOF"],
               s=16, color=colors["2DOF"], edgecolor='0.2', alpha=0.6, zorder=3)

# axes labels/ticks
ax.set_yscale("log")
ax.set_ylabel("BER")
ax.set_xticks(x)
ax.set_xticklabels([f"{int(r)//1000}k" if r>=1000 else f"{int(r)}" for r in rates])
ax.set_xlabel("Data Rate (bps)")
ax.set_xlim(x[0]-0.6, x[-1]+0.6)
ax.set_ylim(1e-5, 1e-1)   # user requested

# right axis: improvement lines — **no legend** entry (label='_nolegend_')
ax2 = ax.twinx()
ax2.plot(x, agg["ImpPID_mean"]*100.0,
         lw=1.4, marker="s", ms=6, color=colors["PID"], label="_nolegend_")
ax2.plot(x, agg["ImpDOF_mean"]*100.0,
         lw=1.4, marker="D", ms=6, color=colors["2DOF"], label="_nolegend_")

# percent annotations (small font)
for i, v in enumerate(agg["ImpDOF_mean"]*100.0):
    ax2.text(x[i]+0.03, v+2.0, f"{v:.1f}%", fontsize=8, color=colors["2DOF"], ha="left", va="bottom")

ax2.set_ylabel("Improvement (%)")
ax2.set_ylim(0, 100)
ax2.set_yticks([0,20,40,60,80,100])

# legend: only bars (compact, inside top-left), fontsize 8, no frame
h_ax, l_ax = ax.get_legend_handles_labels()
leg = ax.legend(h_ax, l_ax,
                loc="upper left",
                bbox_to_anchor=(0.02, 0.98),
                fontsize=8,
                frameon=False,
                ncol=1)
# ensure top margin enough
fig.subplots_adjust(top=0.86)

# spine linewidths
for spine in ax.spines.values():
    spine.set_linewidth(1.0)
for spine in ax2.spines.values():
    spine.set_linewidth(1.0)

# grid and save
ax.grid(True, which="both", axis="y", linestyle=":", alpha=0.25)
fig.tight_layout()
fig.savefig(OUT_PNG, dpi=600, bbox_inches='tight')
fig.savefig(OUT_PDF, bbox_inches='tight')

print(f"Saved: {OUT_PNG} and {OUT_PDF}")
