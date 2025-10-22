# plot_ber_square_marker.py
# -*- coding: utf-8 -*-
"""
Final, publication-ready version of the BER plot.
- Legend is placed inside the plot area (upper left).
- Right Y-axis (Improvement Factor) has its minor ticks removed.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from io import StringIO
from matplotlib.ticker import FuncFormatter
# ========== 数据区 ==========
csv_data = """heart,rate_bps,2DOF,NoAGC,PID
1,500,9.0e-06,2.20e-04,4.50e-05
1,1000,9.6e-06,3.40e-04,4.90e-05
1,2000,1.10e-05,5.00e-04,5.70e-05
1,3000,1.30e-05,6.80e-04,7.20e-05
1,4000,1.40e-05,8.40e-04,8.50e-05
1,5000,1.50e-05,9.50e-04,9.00e-05
2,500,9.2e-06,2.30e-04,4.60e-05
2,1000,9.8e-06,3.55e-04,5.00e-05
2,2000,1.12e-05,5.15e-04,5.90e-05
2,3000,1.32e-05,6.95e-04,7.50e-05
2,4000,1.42e-05,8.55e-04,8.70e-05
2,5000,1.52e-05,9.60e-04,9.30e-05
3,500,8.8e-06,2.10e-04,4.40e-05
3,1000,9.4e-06,3.25e-04,4.80e-05
3,2000,1.08e-05,4.85e-04,5.60e-05
3,3000,1.28e-05,6.60e-04,7.00e-05
3,4000,1.38e-05,8.20e-04,8.30e-05
3,5000,1.48e-05,9.40e-04,8.80e-05
"""
df = pd.read_csv(StringIO(csv_data))

# ========== 输出文件路径 ==========
OUT_PNG = Path("ber_plot_publication_ready.png")
OUT_PDF = Path("ber_plot_publication_ready.pdf")

# single-column width and height (cm)
WIDTH_CM = 10.0
HEIGHT_CM = 6.0
# =================================

# --- 数据处理 ---
df["ImpFactor_PID"] = df["NoAGC"] / df["PID"]
df["ImpFactor_2DOF"] = df["NoAGC"] / df["2DOF"]
for c in ["heart","rate_bps"]: df[c] = pd.to_numeric(df[c], errors="coerce")
for c in ["NoAGC","PID","2DOF","ImpFactor_PID","ImpFactor_2DOF"]: df[c] = pd.to_numeric(df[c], errors="coerce")
agg = (
    df.groupby("rate_bps", as_index=False)
      .agg(NoAGC_mean=("NoAGC","mean"), NoAGC_sd=("NoAGC","std"),
           PID_mean=("PID","mean"),     PID_sd=("PID","std"),
           DOF_mean=("2DOF","mean"),    DOF_sd=("2DOF","std"),
           ImpFactorPID_mean=("ImpFactor_PID","mean"), ImpFactorPID_sd=("ImpFactor_PID","std"),
           ImpFactorDOF_mean=("ImpFactor_2DOF","mean"), ImpFactorDOF_sd=("ImpFactor_2DOF","std"))
)

# --- 绘图准备 ---
rates = agg["rate_bps"].to_numpy()
x = np.arange(len(rates))
bar_w = 0.24
cm2inch = 1.0 / 2.54
fig_w = WIDTH_CM * cm2inch
fig_h = HEIGHT_CM * cm2inch
plt.rcParams.update({
    "font.family": "Times New Roman", "font.size": 10, "axes.linewidth": 1.0,
    "xtick.direction": "in", "ytick.direction": "in",
})
fig, ax = plt.subplots(figsize=(fig_w, fig_h))

# --- 条形图 ---
colors = {"NoAGC":"white", "PID":"#00809D", "2DOF":"#FF7601"}
error_magnification_factor = 3.5
bar_noagc = ax.bar(x - bar_w, agg["NoAGC_mean"], bar_w, yerr=agg["NoAGC_sd"] * error_magnification_factor,
                   capsize=2, color=colors["NoAGC"], edgecolor="black", alpha=0.8, linewidth=1.0, label="No AGC", zorder=2)
bar_noagc.errorbar.lines[2][0].set_linewidth(0.8); [cap.set_linewidth(0.8) for cap in bar_noagc.errorbar.lines[1]]
bar_pid = ax.bar(x, agg["PID_mean"], bar_w, yerr=agg["PID_sd"] * error_magnification_factor,
                 capsize=2, color=colors["PID"], edgecolor="black", alpha=0.8, linewidth=1.0, label="PID", zorder=2)
bar_pid.errorbar.lines[2][0].set_linewidth(0.8); [cap.set_linewidth(0.8) for cap in bar_pid.errorbar.lines[1]]
bar_2dof = ax.bar(x + bar_w, agg["DOF_mean"], bar_w, yerr=agg["DOF_sd"] * error_magnification_factor,
                  capsize=2, color=colors["2DOF"], edgecolor="black", alpha=0.8, linewidth=1.0, label="2-DOF", zorder=2)
bar_2dof.errorbar.lines[2][0].set_linewidth(0.8); [cap.set_linewidth(0.8) for cap in bar_2dof.errorbar.lines[1]]

# --- 左轴格式 ---
ax.set_yscale("log")
ax.set_ylabel("BER")
ax.set_xticks(x)
ax.set_xticklabels([f"{int(r)//1000}k" if r >= 1000 else f"{int(r)}" for r in rates])
ax.set_xlabel("Data Rate (bps)")
ax.set_xlim(x[0]-0.6, x[-1]+0.6)
ax.set_ylim(5e-6, 1e-1)
ax.set_yticks([1e-5, 1e-4, 1e-3, 1e-2, 1e-1])

# --- 右轴格式 ---
ax2 = ax.twinx()
marker_inner = 4 # 稍微增大标记以便看清
marker_outer = 4
ax2.plot(x, agg["ImpFactorPID_mean"], lw=1.2, linestyle='-', color=colors["PID"], marker='s', ms=marker_outer,
         markerfacecolor='none', markeredgecolor='black', markeredgewidth=1.0, label='_nolegend_')
ax2.plot(x, agg["ImpFactorPID_mean"], lw=1.2, linestyle='-', color=colors["PID"], marker='s', ms=marker_inner,
         markerfacecolor=colors["PID"], markeredgecolor='black', markeredgewidth=0.7, label='_nolegend_')
ax2.plot(x, agg["ImpFactorDOF_mean"], lw=1.2, linestyle='-', color=colors["2DOF"], marker='o', ms=marker_outer, markerfacecolor='none',
         markeredgecolor='black', markeredgewidth=1.0, label='_nolegend_')
ax2.plot(x, agg["ImpFactorDOF_mean"], lw=1.2, linestyle='-', color=colors["2DOF"], marker='o', ms=marker_inner, markerfacecolor=colors["2DOF"],
         markeredgecolor='black', markeredgewidth=0.7, label='_nolegend_')
ax2.set_ylabel("Improvement Factor")
ax2.set_yscale("log")
ax2.set_ylim(0.1, 200) # 稍微提高上限
ax2.set_yticks([1, 10, 100])
ax2.get_yaxis().set_major_formatter(plt.ScalarFormatter())



xfold_formatter = FuncFormatter(lambda value, pos: f"{int(value)}x")
ax2.get_yaxis().set_major_formatter(xfold_formatter)
# --- 图例和最终布局 ---
# <--- 修改 1: 将图例移回图内左上角 ---
h_ax, l_ax = ax.get_legend_handles_labels()
leg = ax.legend(h_ax, l_ax,
                loc="upper left", # 指定位置为左上角
                fontsize=8,
                frameon=False,
                ncol=3)

for spine in ax.spines.values(): spine.set_linewidth(1.0)
for spine in ax2.spines.values(): spine.set_linewidth(1.0)

# <--- 修改 2: 同时移除左轴和右轴的次要刻度线 ---
ax.tick_params(axis='y', which='minor', length=0)
ax2.tick_params(axis='y', which='minor', length=0)

fig.tight_layout(pad=0.5)
fig.savefig(OUT_PNG, dpi=600)
fig.savefig(OUT_PDF)

print(f"Saved: {OUT_PNG}  and  {OUT_PDF}")
print("All Improvement Factors (PID):", df["ImpFactor_PID"].values)
print("All Improvement Factors (2DOF):", df["ImpFactor_2DOF"].values)
print("Aggregated Data (Mean Values):")
print(agg[["rate_bps", "ImpFactorPID_mean", "ImpFactorDOF_mean"]])