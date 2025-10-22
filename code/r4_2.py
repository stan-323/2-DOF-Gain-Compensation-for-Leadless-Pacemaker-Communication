# -*- coding: utf-8 -*-
# AGC figure: 1 column × 4 rows, IEEE single-column width (10 cm)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib.ticker as ticker # Import ticker for x-axis intervals

# === 1) 数据读取 ===
CSV_PATH = "your_data.csv"   # ← 改成你的文件路径
df = pd.read_csv(CSV_PATH).apply(pd.to_numeric, errors="coerce")

# 列名映射（与你给的表一致）
# x1, ECG, x2, receiver, x3, Vg, x4, Received
t1, y1 = df["x1"].to_numpy(), df["ECG"].to_numpy()
t2, y2 = df["x2"].to_numpy(), df["receiver"].to_numpy()
t3, y3 = df["x3"].to_numpy(), df["Vg"].to_numpy()
t4, y4 = (df["x4"] / 50.0).to_numpy(), df["Received"].to_numpy()  # 注意：X4÷50

# === 2) 快速数据自检（不会画图，只打印信息）===
def check_series(t, y, name):
    info = {
        "series": name,
        "N": int(np.sum(np.isfinite(t) & np.isfinite(y))),
        "t_min": float(np.nanmin(t)),
        "t_max": float(np.nanmax(t)),
        "y_min": float(np.nanmin(y)),
        "y_max": float(np.nanmax(y)),
        "monotonic_time": bool(np.all(np.diff(t[np.isfinite(t)]) >= 0)),
    }
    print(info)

print("=== Quick sanity check ===")
check_series(t1, y1, "(a) ECG")
check_series(t2, y2, "(b) Rx(no AGC)")
check_series(t3, y3, "(c) Vg control")
check_series(t4, y4, "(d) Rx(with AGC)")

# === 3) 出图（1列4行，IEEE 单栏 10 cm 宽；按你图的样式）===
width_in = 10 / 2.54            # 10 cm = 3.937 in
height_in = 16 / 2.54           # 总高可调，这里 ~16 cm
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({
    "font.size": 8,            # IEEE 8pt 附近，四行更紧凑
    "axes.linewidth": 1,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.direction": "in",
    "ytick.direction": "in",
})

fig, axs = plt.subplots(4, 1, figsize=(width_in, height_in), sharex=False, constrained_layout=True)

# (a) ECG
axs[0].plot(t1, y1, linewidth=1.0)
axs[0].set_ylim(10, 50)
axs[0].set_ylabel("Amplitude (mV)")
axs[0].set_title("ECG Signal", pad=2)
axs[0].text(0.01, 0.97, "(a)", transform=axs[0].transAxes, va="top", ha="left")
axs[0].tick_params(labelbottom=False) # Remove x-axis labels for this subplot

# (b) Received without AGC
axs[1].plot(t2, y2, linewidth=1.0)
axs[1].set_ylim(0, 6)
axs[1].set_ylabel("Amplitude (mV)")
axs[1].set_title("Received Signal Amplitude without AGC", pad=2)
axs[1].text(0.01, 0.97, "(b)", transform=axs[1].transAxes, va="top", ha="left")
axs[1].tick_params(labelbottom=False) # Remove x-axis labels for this subplot

# (c) VGA Control Voltage
axs[2].plot(t3, y3, linewidth=1.0)
axs[2].set_ylim(10, 30)
axs[2].set_ylabel("Vg (mV)")
axs[2].set_title("VGA Control Voltage", pad=2)
axs[2].text(0.01, 0.97, "(c)", transform=axs[2].transAxes, va="top", ha="left")
axs[2].tick_params(labelbottom=False) # Remove x-axis labels for this subplot

# (d) Received with AGC (Compensated) — 时间轴来自 x4/50
axs[3].plot(t4, y4, linewidth=1.0)
axs[3].set_ylim(90, 110)
axs[3].set_xlabel("Time (s)")
axs[3].set_ylabel("Amplitude (mV)")
axs[3].set_title("Received Signal Amplitude with AGC (Compensated)", pad=2)
axs[3].text(0.01, 0.97, "(d)", transform=axs[3].transAxes, va="top", ha="left")

# Set consistent x-axis for all subplots (0 to 3.2)
for ax in axs:
    ax.set_xlim(0, 3.2)
    # Set major ticks every 0.4
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.4))
    # Optional: Set minor ticks every 0.2
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.2))


# Save (600 dpi), consistent with the paper, can export PNG/PDF
out_dir = Path("fig_out"); out_dir.mkdir(parents=True, exist_ok=True)
fig.savefig(out_dir / "agc_1col4rows_ieee_modified.png", dpi=600, bbox_inches="tight")
#fig.savefig(out_dir / "agc_1col4rows_ieee_modified.pdf", bbox_inches="tight")

# plt.show() # Display the plot