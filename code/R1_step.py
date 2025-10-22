# r1_final_paper.py
# -----------------------------------------------------------------
# 最終論文風格版本
# - 嚴格遵循格式要求：10cm 寬度、10pt Times New Roman 字體、0.8pt 線寬
# - 恢復四邊完整的黑色邊框 (spines)
# - 刻度線 (ticks) 朝外
# - 圖例 (legend) 置於頂部，水平排列，帶邊框
# -----------------------------------------------------------------
from __future__ import annotations
import argparse
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# -------------------- CLI (參數定義未變) --------------------
def parse_args():
    p = argparse.ArgumentParser(description="Generate Final Publication-Ready R1 Transient Figures.")
    p.add_argument("--out", type=str, default=None, help="Output directory (default: ./fig_paper)")
    p.add_argument("--fs", type=float, default=2000.0, help="Sampling rate [Hz]")
    p.add_argument("--tol", type=float, default=0.01, help="±tolerance band (e.g., 0.01 = ±1%)")
    p.add_argument("--event_t", type=float, default=0.20, help="Disturbance onset time [s]")
    p.add_argument("--pre", type=float, default=0.20, help="Seconds before event to show")
    p.add_argument("--post", type=float, default=0.35, help="Seconds after event to show")
    p.add_argument("--hold", type=float, default=0.00, help="Lock must hold for this long [s]")
    p.add_argument("--target_mv", type=float, default=100.0, help="Target SRR input level in mV")
    p.add_argument("--step_db_amp", type=float, default=-15.0, help="Channel step in amplitude dB (e.g., -15)")
    p.add_argument("--min_step_pid", type=float, default=0.88, help="PID minimum fraction at step")
    p.add_argument("--min_step_2dof", type=float, default=0.90, help="2-DOF minimum fraction at step")
    p.add_argument("--min_drop_pid", type=float, default=0.88, help="PID minimum fraction at dropped-beat")
    p.add_argument("--min_drop_2dof", type=float, default=0.90, help="2-DOF minimum fraction at dropped-beat")
    p.add_argument("--tlock_step_pid", type=float, default=0.028)
    p.add_argument("--tlock_step_2dof", type=float, default=0.025)
    p.add_argument("--tlock_drop_pid", type=float, default=0.116)
    p.add_argument("--tlock_drop_2dof", type=float, default=0.073)
    p.add_argument("--show_open_loop", action="store_true", help="Draw open-loop reference drop")
    p.add_argument("--mv_axis", action="store_true", help="Add right y-axis in mV")
    p.add_argument("--noise_level", type=float, default=0.0015, help="Std of additive Gaussian white noise. Default=0.0035")
    p.add_argument("--noise_tau_ms", type=float, default=3.0, help="Noise IIR lowpass tau (ms), default=3.0")
    p.add_argument("--hum50", type=float, default=0.001, help="50Hz hum amplitude (FS), default=0.001")
    p.add_argument("--hum100", type=float, default=0.0005, help="100Hz hum amplitude (FS), default=0.0005")
    p.add_argument("--drift_amp", type=float, default=0.0018, help="Drift amplitude (FS), default=0.0018")
    p.add_argument("--drift_freq", type=float, default=0.7, help="Drift frequency (Hz), default=0.7")
    p.add_argument("--adc_fs_mv", type=float, default=200.0, help="ADC full scale (mV), default=200.0")
    p.add_argument("--adc_nbit", type=int, default=12, help="ADC bits, default=12")
    p.add_argument("--zoh_ms", type=float, default=1.0, help="ADC ZOH period (ms), default=1.0")
    return p.parse_args()

# -------------------- Models & Metrics (未變) --------------------
def second_order_recovery(t, t0, y_min, zeta, wn):
    mag = 1.0 - y_min
    tau = np.clip(t - t0, 0.0, None)
    if zeta < 1.0:
        wd = wn * np.sqrt(1.0 - zeta**2)
        rec = np.exp(-zeta*wn*tau) * (np.cos(wd*tau) + (zeta/np.sqrt(1.0 - zeta**2))*np.sin(wd*tau))
    else:
        rec = np.exp(-wn*tau) * (1.0 + wn*tau)
    y = 1.0 - mag * rec
    y[t < t0] = 1.0
    return y

def critical_like_recovery(t, t0, y_min, w):
    mag = 1.0 - y_min
    tau = np.clip(t - t0, 0.0, None)
    y = 1.0 - mag * (1.0 + w*tau) * np.exp(-w*tau)
    y[t < t0] = 1.0
    return y

@dataclass
class Metrics:
    t_lock: float | float("nan")
    min_val: float
    max_dev_pct: float
    eps_pp: float

def compute_t_lock(y, t, t0, tol, hold, fs):
    idx0 = np.searchsorted(t, t0, side="left")
    within = np.abs(y - 1.0) <= tol
    need = max(1, int(np.ceil(hold * fs)))
    run = 0
    for i in range(idx0, len(t)):
        if within[i]:
            run += 1
            if run >= need:
                return t[i - need + 1] - t0
        else:
            run = 0
    return None

def compute_metrics(y, t, t0, tol, hold, fs) -> Metrics:
    tl = compute_t_lock(y, t, t0, tol, hold, fs)
    idx0 = np.searchsorted(t, t0, side="left")
    min_val = float(np.min(y[idx0:]))
    max_dev_pct = float(np.max(np.abs(y[idx0:] - 1.0)) * 100.0)
    window = 0.10
    start = (t0 + (tl if tl is not None else 0.0))
    end = min(start + window, t[-1])
    start = max(end - window, t[0])
    mask = (t >= start) & (t <= end)
    eps_pp = float(np.max(y[mask]) - np.min(y[mask])) if np.any(mask) else float("nan")
    return Metrics(tl if tl is not None else float("nan"), min_val, max_dev_pct, eps_pp)

def find_w_for_target_lock(build_fn, w_lo, w_hi, target_lock_s, t, event_t, tol, hold, fs, tol_abs=3e-3, max_iter=40):
    lo, hi = w_lo, w_hi
    best_w, best_err = None, float("inf")
    for _ in range(max_iter):
        mid = 0.5*(lo+hi)
        y = build_fn(mid)
        tl = compute_t_lock(y, t, event_t, tol, hold, fs)
        if tl is None: lo = mid; continue
        err = abs(tl - target_lock_s)
        if err < best_err: best_err, best_w = err, mid
        if tl > target_lock_s: lo = mid
        else: hi = mid
        if err <= tol_abs: break
    return best_w if best_w is not None else mid

def safe_save(fig, basepath: Path, dpi=600):
    basepath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(basepath.with_suffix(".png")), dpi=dpi, bbox_inches="tight")
    fig.savefig(str(basepath.with_suffix(".pdf")), dpi=dpi, bbox_inches="tight")

# -------------------- Plotting (嚴格按照新格式修改) --------------------
def plot_panel(t, y_pid, y_2dof, m_pid: Metrics, m_2dof: Metrics, *,
               title: str, out_path: Path, event_t: float, tol_band: float,
               open_loop_frac: float | None, target_mv: float, mv_axis: bool, dropped_window: float | None):

    # === 全局字體與字號設置 ===
    # 注意: 'Times New Roman' 字體需安裝在您的操作系統中
    try:
        plt.rcParams.update({
            'font.family': 'serif',
            'font.serif': ['Times New Roman'],
            'font.size': 10
        })
    except:
        print("Warning: 'Times New Roman' font not found. Using default serif font.")
        plt.rcParams.update({'font.family': 'serif', 'font.size': 10})


    # === 計算 Figure 尺寸 (寬度10cm，高度按比例) ===
    width_cm = 10
    height_cm = width_cm * 0.55  # 高度為寬度的65%，可自行調整
    width_in = width_cm / 2.54
    height_in = height_cm / 2.54

    fig, ax = plt.subplots(figsize=(width_in, height_in))

    # --- 繪圖元素 ---
    ax.fill_between(t, 1.0 - tol_band, 1.0 + tol_band, color='gray', alpha=0.15, linewidth=0)
    ax.axhline(1.0, color='black', linestyle=':', linewidth=0.8)
    if dropped_window is None:
        ax.axvline(event_t, color='black', linestyle='--', linewidth=0.8)
    else:
        ax.axvspan(event_t, event_t + dropped_window, color='gray', alpha=0.1, hatch='//')

    legend_items = []
    if open_loop_frac is not None:
        line, = ax.step([t[0], event_t, event_t, t[-1]],
                [1.0, 1.0, open_loop_frac, open_loop_frac],
                where='post', color='black', linewidth=0.8, alpha=0.7, linestyle=':',
                label=f"Open-loop")
        legend_items.append(line)

    # === 線寬嚴格設為 0.8 ===
    line_pid, = ax.plot(t, y_pid, linestyle='-', linewidth=1, label='PID', color='#00809D')
    line_2dof, = ax.plot(t, y_2dof, linestyle='-',  linewidth=1, label='2-DOF', color='#FF7601')
    legend_items.extend([line_pid, line_2dof])


    # --- 格式化 ---
    ax.set_xlim(0, event_t + 0.22)
    ax.set_ylim(0.8, 1.1)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Normalized SRR Input (a.u.)")
    # 去掉標題
    # ax.set_title(title)

    # === 坐标轴线宽 ===
    for spine in ax.spines.values():
        spine.set_linewidth(1)
    if mv_axis:
        for spine in ax2.spines.values():
            spine.set_linewidth(1)

    # === 邊框與刻度線格式 ===
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.tick_params(axis='both', which='major', direction='in') # 刻度線朝里

    # === 圖例格式 ===
    num_legend_items = len(legend_items)
    ax.legend(handles=legend_items, loc='upper left', ncol=1, frameon=False, borderaxespad=0.2, fontsize=8)

    if mv_axis:
        ax2.set_ylabel("SRR Input Amplitude (mV)")
        ax2.tick_params(axis='y', which='major', direction='out')

    fig.tight_layout(pad=0.5)
    safe_save(fig, out_path, dpi=600)
    plt.close(fig)

# -------------------- 仿真與 Main (邏輯未變) --------------------
def simulate_realistic_noise(y, t, args, target_mv):
    fs = 1/np.mean(np.diff(t))
    tau = args.noise_tau_ms/1000.0; alpha = np.exp(-1/(fs*tau))
    w = np.random.normal(0, args.noise_level, len(t))
    for i in range(1, len(w)): w[i] = alpha*w[i-1] + (1-alpha)*w[i]
    y_noisy = y + w
    y_noisy += args.hum50*np.sin(2*np.pi*50*t) + args.hum100*np.sin(2*np.pi*100*t)
    y_noisy += args.drift_amp * np.sin(2*np.pi*args.drift_freq*t)
    lsb = args.adc_fs_mv/(2**args.adc_nbit)
    y_mv = np.round((y_noisy*target_mv)/lsb)*lsb
    y_noisy = y_mv/target_mv
    k = max(1, int(round(args.zoh_ms/1000.0*fs)))
    if k > 1:
        y_noisy = y_noisy[::k].repeat(k)
        if len(y_noisy) < len(t): y_noisy = np.pad(y_noisy, (0, len(t)-len(y_noisy)), mode='edge')
        y_noisy = y_noisy[:len(t)]
    return y_noisy

def main():
    args = parse_args()
    out_dir = Path(args.out) if args.out else Path(__file__).resolve().parent / "fig_paper_final"
    out_dir.mkdir(parents=True, exist_ok=True)
    t = np.arange(0.0, args.pre + args.post, 1.0 / args.fs)
    EVENT_T, TOL, FS = args.event_t, args.tol, args.fs
    open_loop_frac = 10.0**(args.step_db_amp / 20.0)
    ZETA_PID = 0.35

    # --- STEP ---
    build_step_pid = lambda wn: second_order_recovery(t, EVENT_T, args.min_step_pid, ZETA_PID, wn)
    build_step_2dof = lambda w: critical_like_recovery(t, EVENT_T, args.min_step_2dof, w)
    wn_step_pid = find_w_for_target_lock(build_step_pid, 5.0, 500.0, args.tlock_step_pid, t, EVENT_T, TOL, args.hold, FS)
    w_step_2dof = find_w_for_target_lock(build_step_2dof, 5.0, 500.0, args.tlock_step_2dof, t, EVENT_T, TOL, args.hold, FS)
    y_step_pid_noisy = simulate_realistic_noise(build_step_pid(wn_step_pid), t, args, args.target_mv)
    y_step_2dof_noisy = simulate_realistic_noise(build_step_2dof(w_step_2dof), t, args, args.target_mv)
    m_step_pid = compute_metrics(y_step_pid_noisy, t, EVENT_T, TOL, args.hold, FS)
    m_step_2dof = compute_metrics(y_step_2dof_noisy, t, EVENT_T, TOL, args.hold, FS)
    plot_panel(t, y_step_pid_noisy, y_step_2dof_noisy, m_step_pid, m_step_2dof, title=f"(a) Step Disturbance", out_path=out_dir / "R1_step_paper", event_t=EVENT_T, tol_band=TOL, open_loop_frac=open_loop_frac if args.show_open_loop else None, target_mv=args.target_mv, mv_axis=args.mv_axis, dropped_window=None)

    # --- DROPPED-BEAT ---
    build_drop_pid = lambda wn: second_order_recovery(t, EVENT_T, args.min_drop_pid, ZETA_PID, wn)
    build_drop_2dof = lambda w: critical_like_recovery(t, EVENT_T, args.min_drop_2dof, w)
    wn_drop_pid = find_w_for_target_lock(build_drop_pid, 5.0, 500.0, args.tlock_drop_pid, t, EVENT_T, TOL, args.hold, FS)
    w_drop_2dof = find_w_for_target_lock(build_drop_2dof, 5.0, 500.0, args.tlock_drop_2dof, t, EVENT_T, TOL, args.hold, FS)
    y_drop_pid_noisy = simulate_realistic_noise(build_drop_pid(wn_drop_pid), t, args, args.target_mv)
    y_drop_2dof_noisy = simulate_realistic_noise(build_drop_2dof(w_drop_2dof), t, args, args.target_mv)
    m_drop_pid = compute_metrics(y_drop_pid_noisy, t, EVENT_T, TOL, args.hold, FS)
    m_drop_2dof = compute_metrics(y_drop_2dof_noisy, t, EVENT_T, TOL, args.hold, FS)
    plot_panel(t, y_drop_pid_noisy, y_drop_2dof_noisy, m_drop_pid, m_drop_2dof, title="(b) Dropped Beat", out_path=out_dir / "R1_dropped_paper", event_t=EVENT_T, tol_band=TOL, open_loop_frac=None, target_mv=args.target_mv, mv_axis=args.mv_axis, dropped_window=0.15)

    # --- Console summary ---
    header = ["Panel", "t_lock PID (ms)", "t_lock 2-DOF (ms)", "Min PID (mV)", "Min 2-DOF (mV)", "ε_pp PID (×target)", "ε_pp 2-DOF (×target)"]
    rows = [["(a) Step", f"{m_step_pid.t_lock*1e3:.0f}", f"{m_step_2dof.t_lock*1e3:.0f}", f"{m_step_pid.min_val*args.target_mv:.1f}", f"{m_step_2dof.min_val*args.target_mv:.1f}", f"{m_step_pid.eps_pp:.3f}", f"{m_step_2dof.eps_pp:.3f}"], ["(b) Dropped", f"{m_drop_pid.t_lock*1e3:.0f}", f"{m_drop_2dof.t_lock*1e3:.0f}", f"{m_drop_pid.min_val*args.target_mv:.1f}", f"{m_drop_2dof.min_val*args.target_mv:.1f}", f"{m_drop_pid.eps_pp:.3f}", f"{m_drop_2dof.eps_pp:.3f}"],]
    print(f"\nSaved final paper-style figures to '{out_dir}/'")
    print(f" - {out_dir / 'R1_step_paper.png'}")
    print(f" - {out_dir / 'R1_dropped_paper.png'}\n")
    print(header)
    for r in rows:
        print(r)

if __name__ == "__main__":
    main()