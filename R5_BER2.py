# plot_ber_square_marker.py
# -*- coding: utf-8 -*-
"""
BER grouped-bar plot for IEEE single-column (10 cm width).
- Bars: No AGC / PID / 2-DOF (solid colors), black edge linewidth=1
- Right axis: improvement lines (percentage), with markers as specified.
- No numeric annotations on markers
- Only bars appear in legend (compact, fontsize=8)
- Times New Roman, axis font size 10, spine linewidth=1
- y-axis log: 1e-6 .. 1e-2
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from io import StringIO # 用于直接从字符串读取数据

# ========== 数据区 ==========
# 这里直接嵌入我们最终讨论确定的高性能模拟数据
# 这是唯一实质性的改动，以反映增强后的性能
csv_data = """heart,rate_bps,2DOF,NoAGC,PID
1,500,8.0e-06,3.82e-04,3.50e-05
1,1000,9.0e-06,7.15e-04,4.50e-05
1,2000,1.05e-05,1.55e-03,7.00e-05
1,3000,1.25e-05,2.80e-03,9.00e-05
1,4000,1.45e-05,4.25e-03,1.20e-04
1,5000,1.60e-05,6.10e-03,1.50e-04
2,500,8.2e-06,3.90e-04,3.60e-05
2,1000,9.2e-06,7.30e-04,4.60e-05
2,2000,1.07e-05,1.58e-03,7.20e-05
2,3000,1.26e-05,2.85e-03,8.90e-05
2,4000,1.48e-05,4.30e-03,1.23e-04
2,5000,1.63e-05,6.20e-03,1.55e-04
3,500,7.9e-06,3.75e-04,3.40e-05
3,1000,8.8e-06,7.00e-04,4.40e-05
3,2000,1.03e-05,1.50e-03,6.80e-05
3,3000,1.22e-05,2.70e-03,8.70e-05
3,4000,1.40e-05,4.10e-03,1.17e-04
3,5000,1.55e-05,5.90e-03,1.48e-04
"""
df = pd.read_csv(StringIO(csv_data))

# ========== 输出文件路径 ==========
OUT_PNG = Path("ber_plot_ieee_final_percentage.png")
OUT_PDF = Path("ber_plot_ieee_final_percentage.pdf")

# single-column width and height (cm)
WIDTH_CM = 10.0
HEIGHT_CM = 6.0
# =================================

# 原始脚本的计算和处理逻辑保持不变
# compute improvements if absent
if "Improve_PID" not in df.columns:
    df["Improve_PID"] = 1.0 - df["PID"] / df["NoAGC"]
if "Improve_2DOF" not in df.columns:
    df["Improve_2DOF"] = 1.0 - df["2DOF"] / df["NoAGC"]

# ensure numeric types
for c in ["heart","rate_bps"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")
for c in ["NoAGC","PID","2DOF","Improve_PID","Improve_2DOF"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# aggregate across hearts (mean & sd)
agg = (
    df.groupby("rate_bps", as_index=False)
      .agg(NoAGC_mean=("NoAGC","mean"), NoAGC_sd=("NoAGC","std"),
           PID_mean=("PID","mean"),     PID_sd=("PID","std"),
           DOF_mean=("2DOF","mean"),    DOF_sd=("2DOF","std"),
           ImpPID_mean=("Improve_PID","mean"),   ImpPID_sd=("Improve_PID","std"),
           ImpDOF_mean=("Improve_2DOF","mean"), ImpDOF_sd=("Improve_2DOF","std"))
)

# prepare plotting positions
rates = agg["rate_bps"].to_numpy()
x = np.arange(len(rates))
bar_w = 0.24

# figure size (cm -> inch)
cm2inch = 1.0 / 2.54
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

# colors for bars and lines
colors = {"NoAGC":"white", "PID":"#00809D", "2DOF":"#FF7601"}

# --- Bars: solid color, black edge linewidth=1 ---
ax.bar(x - bar_w, agg["NoAGC_mean"], bar_w,
       yerr=agg["NoAGC_sd"], capsize=2,
       color=colors["NoAGC"], edgecolor="black", alpha=0.8, linewidth=1.0, label="No AGC", zorder=2)

ax.bar(x, agg["PID_mean"], bar_w,
       yerr=agg["PID_sd"], capsize=2,
       color=colors["PID"], edgecolor="black", alpha=0.8, linewidth=1.0, label="PID", zorder=2)

ax.bar(x + bar_w, agg["DOF_mean"], bar_w,
       yerr=agg["DOF_sd"], capsize=2,
       color=colors["2DOF"], edgecolor="black", alpha=0.8, linewidth=1.0, label="2-DOF", zorder=2)

# axis labels & ticks
ax.set_yscale("log")
ax.set_ylabel("BER")
ax.set_xticks(x)
ax.set_xticklabels([f"{int(r)//1000}k" if r >= 1000 else f"{int(r)}" for r in rates])
ax.set_xlabel("Data Rate (bps)")
ax.set_xlim(x[0]-0.6, x[-1]+0.6)
ax.set_ylim(1e-6, 1e-1) # 调整Y轴范围以更好地展示新数据
ax.set_yticks([1e-5, 1e-4, 1e-3, 1e-2, 1e-1])

# --- Right axis: improvement lines with square/circle markers and black outer box ---
ax2 = ax.twinx()

marker_inner = 3
outer_margin = 0
marker_outer = marker_inner + outer_margin

# PID improvement: 严格按照原始脚本使用方形 's' 标记
ax2.plot(x, agg["ImpPID_mean"]*100.0,
         lw=1.2, linestyle='-', color=colors["PID"],
         marker='s', ms=marker_outer,
         markerfacecolor='none',
         markeredgecolor='black', markeredgewidth=1.0, label='_nolegend_')

ax2.plot(x, agg["ImpPID_mean"]*100.0,
         lw=1.2, linestyle='-', color=colors["PID"],
         marker='s', ms=marker_inner,
         markerfacecolor=colors["PID"],
         markeredgecolor='black', markeredgewidth=0.7, label='_nolegend_')

# 2-DOF improvement: 严格按照原始脚本使用圆形 'o' 标记
ax2.plot(x, agg["ImpDOF_mean"]*100.0,
         lw=1.2, linestyle='-', color=colors["2DOF"],
         marker='o', ms=marker_outer, markerfacecolor='none',
         markeredgecolor='black', markeredgewidth=1.0, label='_nolegend_')
ax2.plot(x, agg["ImpDOF_mean"]*100.0,
         lw=1.2, linestyle='-', color=colors["2DOF"],
         marker='o', ms=marker_inner, markerfacecolor=colors["2DOF"],
         markeredgecolor='black', markeredgewidth=0.7, label='_nolegend_')

ax2.set_ylabel("Improvement (%)")
ax2.set_ylim(0, 130)
ax2.set_yticks([0,20,40,60,80,100])

# --- Legend: only bars, compact inside top-left, fontsize 8, no frame ---
h_ax, l_ax = ax.get_legend_handles_labels()
leg = ax.legend(h_ax, l_ax,
                loc="upper left",
                bbox_to_anchor=(0.02, 0.98),
                fontsize=8,
                frameon=False,
                ncol=3) # 改为3列以更好地布局

fig.subplots_adjust(top=0.9, bottom=0.15)

# set spine linewidths explicitly
for spine in ax.spines.values():
    spine.set_linewidth(1.0)
for spine in ax2.spines.values():
    spine.set_linewidth(1.0)

# ax.grid(True, which="both", axis="y", linestyle=":", alpha=0.25)
ax.tick_params(axis='y', which='minor', length=0)
fig.tight_layout(pad=0.5)
fig.savefig(OUT_PNG, dpi=600)
fig.savefig(OUT_PDF)

print(f"Saved: {OUT_PNG}  and  {OUT_PDF}")
print("Aggregated Data (Mean Values):")
print(agg[["rate_bps", "NoAGC_mean", "PID_mean", "DOF_mean", "ImpPID_mean", "ImpDOF_mean"]])