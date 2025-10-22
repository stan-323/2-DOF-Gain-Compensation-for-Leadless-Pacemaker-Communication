# r4_robustness.py
# -------------------------------------------------
# Robustness to feedforward mismatches (δt, δA)
# Main: heat/contour map + feasible region
# Extra: BER slices vs δt (fixed δA) & vs δA (fixed δt)
# matplotlib only; one chart per figure; no custom colors
# -------------------------------------------------

import numpy as np, matplotlib.pyplot as plt
from pathlib import Path

# ===== Config =====
dt_range = (-80.0, 80.0)   # ms
da_range = (-6.0, 6.0)     # dB
n_dt, n_da = 121, 121

RES_THRESH = 0.01          # residual RMS < 1% (fraction of target)
BER_THRESH = 1e-3

# 简单理想化模型参数（调这些能“放大/缩小”可行岛）
K_A = 0.02   # 振幅失配 → 残差的比例系数
K_T = 0.01   # 定时失配 → 残差的比例系数（相对 40 ms）
T_REF = 40.0 # ms

BER_MIN  = 3e-4  # 良性区的 BER 底值
BER_GAIN = 1.0   # 残差→BER 的增益
BER_EXP  = 2.5   # 指数

# 切片（右边两张图）
slice_da_values = [0.0, +3.0, -3.0]   # dB
slice_dt_values = [0.0, +40.0, -40.0] # ms

# ===== Grid & model =====
dt = np.linspace(*dt_range, n_dt)
da = np.linspace(*da_range, n_da)
DT, DA = np.meshgrid(dt, da, indexing='xy')

A  = 10**(DA/20.0)               # 振幅线性比
eA = K_A * np.abs(A - 1.0)       # 振幅导致的残差分量
eT = K_T * np.abs(DT) / T_REF    # 定时导致的残差分量
RES = np.sqrt(eA**2 + eT**2)     # 合成残差（分数）

# 理想化 BER 映射（单调递增）
BER = BER_MIN * (1.0 + BER_GAIN * (RES/RES_THRESH)**BER_EXP)

mask = (RES <= RES_THRESH) & (BER <= BER_THRESH)

# ===== Plot 1: robustness map =====
out_dir = Path.cwd() / "fig"; out_dir.mkdir(parents=True, exist_ok=True)
fig, ax = plt.subplots(figsize=(8.6, 6.2))

im = ax.contourf(DT, DA, RES, levels=30)            # 残差热力
ax.contour(DT, DA, RES, levels=[RES_THRESH], linewidths=1.6)              # 残差阈值
ax.contour(DT, DA, BER, levels=[BER_THRESH], linewidths=1.6, linestyles='--')  # BER 阈值

# 可行区域淡遮罩
feas = np.where(mask, 1.0, np.nan)
ax.pcolormesh(DT, DA, feas, shading='auto', alpha=0.20)

ax.scatter([0],[0], marker='*', s=120, label="This work (0,0)")
ax.set_xlabel("Timing mismatch δt (ms)")
ax.set_ylabel("Amplitude mismatch δA (dB)")
ax.set_title("R4 — Robustness map: residual & BER thresholds")
ax.grid(True, linestyle=':', linewidth=0.8, alpha=0.6)
ax.legend(loc='upper right', fontsize=9)
fig.tight_layout()
base = out_dir/"R4_robustness_map"
fig.savefig(str(base.with_suffix(".png")), dpi=300, bbox_inches="tight")
fig.savefig(str(base.with_suffix(".pdf")), dpi=300, bbox_inches="tight")

# ===== Plot 2: BER vs δt (fixed δA) =====
fig2, ax2 = plt.subplots(figsize=(8.6, 4.8))
for da_fix in slice_da_values:
    A_fix = 10**(da_fix/20.0)
    eA_fix = K_A * abs(A_fix - 1.0)
    eT_line = K_T * np.abs(dt) / T_REF
    res_line = np.sqrt(eA_fix**2 + eT_line**2)
    ber_line = BER_MIN * (1.0 + BER_GAIN * (res_line/RES_THRESH)**BER_EXP)
    ax2.plot(dt, ber_line, label=f"δA={da_fix:+.0f} dB")
ax2.axhline(BER_THRESH, linestyle='--', linewidth=1.2)
ax2.set_yscale('log'); ax2.set_ylim(min(BER_MIN*0.8,5e-5), max(BER_THRESH*3,2e-2))
ax2.set_xlabel("Timing mismatch δt (ms)")
ax2.set_ylabel("BER (log)")
ax2.set_title("R4 — BER vs δt (slices at fixed δA)")
ax2.grid(True, which='both', linestyle=':', linewidth=0.8, alpha=0.6)
ax2.legend(loc='upper left', fontsize=9)
fig2.tight_layout()
base2 = out_dir/"R4_slices_vs_dt"
fig2.savefig(str(base2.with_suffix(".png")), dpi=300, bbox_inches="tight")
fig2.savefig(str(base2.with_suffix(".pdf")), dpi=300, bbox_inches="tight")

# ===== Plot 3: BER vs δA (fixed δt) =====
fig3, ax3 = plt.subplots(figsize=(8.6, 4.8))
for dt_fix in slice_dt_values:
    eT_fix = K_T * abs(dt_fix) / T_REF
    A_line = 10**(da/20.0)
    eA_line = K_A * np.abs(A_line - 1.0)
    res_line = np.sqrt(eA_line**2 + eT_fix**2)
    ber_line = BER_MIN * (1.0 + BER_GAIN * (res_line/RES_THRESH)**BER_EXP)
    ax3.plot(da, ber_line, label=f"δt={dt_fix:+.0f} ms")
ax3.axhline(BER_THRESH, linestyle='--', linewidth=1.2)
ax3.set_yscale('log'); ax3.set_ylim(min(BER_MIN*0.8,5e-5), max(BER_THRESH*3,2e-2))
ax3.set_xlabel("Amplitude mismatch δA (dB)")
ax3.set_ylabel("BER (log)")
ax3.set_title("R4 — BER vs δA (slices at fixed δt)")
ax3.grid(True, which='both', linestyle=':', linewidth=0.8, alpha=0.6)
ax3.legend(loc='upper left', fontsize=9)
fig3.tight_layout()
base3 = out_dir/"R4_slices_vs_da"
fig3.savefig(str(base3.with_suffix(".png")), dpi=600, bbox_inches="tight")
fig3.savefig(str(base3.with_suffix(".pdf")), dpi=600, bbox_inches="tight")
