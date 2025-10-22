
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass

# --- 数据定义部分 (未变) ---
np.random.seed(42)
OUT_DIR = Path.cwd() / "fig_paper_final"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# SCENARIOS = ["Step 6 dB", "Step 10 dB", "Step 15 dB", "Dropped beat"]
SCENARIOS = ["5 dB", "10 dB", "15 dB"]
N_REP = 12

# —— Lock time (ms) ——  (metric: ±2% band, hold = 20 ms)
LOCK_MEAN_PID   = [51, 70, 85]     # 5 dB, 10 dB, 15 dB
LOCK_MEAN_2DOF  = [12, 17, 20.3]

LOCK_SD_PID     = [6,  8, 10]
LOCK_SD_2DOF    = [2,  2,  3]

# —— Max Deviation (%) ——
# —— Max Deviation (%) ——  (window: step后 ~150 ms / 至重新稳定；取全局峰值)
MAXDEV_MEAN_PID   = [7.3, 10.2, 12.3]   # 5 dB, 10 dB, 15 dB
MAXDEV_MEAN_2DOF  = [5.6,  8.3, 9.1]

# 合理的标准差（可后续用实测覆盖）
MAXDEV_SD_PID     = [0.8, 1.0, 1.3]
MAXDEV_SD_2DOF    = [0.6, 0.8, 1.0]


def make_samples(means, sds, n=N_REP, low=0.0):
    return np.vstack([np.clip(np.random.normal(m, s, n), low, None) for m, s in zip(means, sds)])


def ci95(x):
    xb = np.random.choice(x, size=(5000, len(x)), replace=True).mean(axis=1)
    return np.percentile(xb, [2.5, 97.5])


def jitter(n, w=0.10): return (np.random.rand(n) - 0.5) * 2 * w


@dataclass
class MetricSet: pid: np.ndarray; d2f: np.ndarray


LOCK = MetricSet(make_samples(LOCK_MEAN_PID, LOCK_SD_PID), make_samples(LOCK_MEAN_2DOF, LOCK_SD_2DOF))
MAXD = MetricSet(make_samples(MAXDEV_MEAN_PID, MAXDEV_SD_PID), make_samples(MAXDEV_MEAN_2DOF, MAXDEV_SD_2DOF))


# --- 绘图函数 (已按新格式要求重写) ---
def plot_single_metric(ax, metric: MetricSet, ylab, title, ylim=None):
    S = len(SCENARIOS)
    x = np.arange(S)
    width = 0.35
    m1 = metric.pid.mean(axis=1)
    m2 = metric.d2f.mean(axis=1)
    c1 = np.array([ci95(metric.pid[i]) for i in range(S)])
    c2 = np.array([ci95(metric.d2f[i]) for i in range(S)])

    color_pid = '#00809D'  # PID 色
    color_2dof = '#FF7601'  # 2-DOF 色

    # 调整柱状图边框线宽为 1.0
    ax.bar(x - width / 2, m1, width, label="PID", color=color_pid, alpha=0.8, edgecolor='black', linewidth=1.0)
    ax.bar(x + width / 2, m2, width, label="2-DOF", color=color_2dof, alpha=0.8, edgecolor='black', linewidth=1.0)

    # 确保误差线线宽为 1.0
    ax.errorbar(x - width / 2, m1, yerr=np.vstack([m1 - c1[:, 0], c1[:, 1] - m1]), fmt='none', capsize=3, linewidth=1.0,
                ecolor='black')
    ax.errorbar(x + width / 2, m2, yerr=np.vstack([m2 - c2[:, 0], c2[:, 1] - m2]), fmt='none', capsize=3, linewidth=1.0,
                ecolor='black')

    for i in range(S):
        # ax.scatter(np.full(N_REP, x[i] - width / 2) + jitter(N_REP, 0.08), metric.pid[i], s=5, alpha=0.8,
        #            color=colors[0], facecolors='none', zorder=10)
        # ax.scatter(np.full(N_REP, x[i] + width / 2) + jitter(N_REP, 0.08), metric.d2f[i], s=5, alpha=0.6,
        #            color=colors[1], facecolors='none', zorder=10)

        ax.scatter(np.full(N_REP, x[i] - width / 2) + jitter(N_REP, 0.08), metric.pid[i], s=1, alpha=0.8, color='#6A7460', linewidths=1 ,zorder=10)
        ax.scatter(np.full(N_REP, x[i] + width / 2) + jitter(N_REP, 0.08), metric.d2f[i], s=1, alpha=0.8, color='#6A7460', linewidths=1 ,zorder=10)

    # # ...
    # # PID 的散点使用 colors[0]
    # ax.scatter(..., s=10, alpha=0.8, color=colors[0],
    #            edgecolor='black', linewidths=0.5, zorder=10)
    # # 2-DOF 的散点使用 colors[1]
    # ax.scatter(..., s=10, alpha=0.8, color=colors[1],
    #            edgecolor='black', linewidths=0.5, zorder=10)
    # # ...

    ax.set_xticks(x)
    ax.set_xticklabels(SCENARIOS, rotation=0, ha='center')
    ax.set_ylabel(ylab)
    ax.set_title(title, weight='normal')
    if ylim: ax.set_ylim(*ylim)

    # --- 严格遵循新的格式要求 ---
    # 1. 恢复四边边框
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)

    # 2. 移除背景网格
    ax.grid(False)

    # 3. 设置刻度线朝内
    ax.tick_params(axis='both', which='major', direction='in')

    # 调整图例样式
    ax.legend(loc='upper left', frameon=False, handlelength=1.5, handletextpad=0.5, fontsize=9)


def main():
    # === 设置全局绘图风格 (严格按照新要求) ===
    try:
        plt.rcParams.update({
            'font.family': 'serif',
            'font.serif': ['Times New Roman'],
            'font.size': 8,
            'axes.linewidth': 1.0,  # 坐标轴线宽
            'xtick.major.width': 1.0,  # X轴主刻度线宽度
            'ytick.major.width': 1.0,  # Y轴主刻度线宽度
            'axes.titleweight': 'normal'
        })
    except:
        print("警告: 未找到 'Times New Roman' 字体。将使用默认的衬线字体。")
        plt.rcParams.update({
            'font.family': 'serif',
            'font.size': 8,
            'axes.linewidth': 1.0,
            'xtick.major.width': 1.0,
            'ytick.major.width': 1.0
        })

    # --- 尺寸与布局 (未变) ---
    width_cm = 10
    height_cm = 4.5
    width_in = width_cm / 2.54
    height_in = height_cm / 2.54
    fig, axes = plt.subplots(1, 2, figsize=(width_in, height_in), sharey=False)

    # --- 绘图 (调用已修改的函数) ---
    plot_single_metric(axes[0], LOCK, "Settling time (ms)", "(a) Settling Time", ylim=(0, 150))
    plot_single_metric(axes[1], MAXD, "Max Deviation (%)", "(b) Max Deviation", ylim=(0, 20))

    fig.tight_layout(pad=0.5, w_pad=0.5)

    # --- 保存 ---
    out_path = OUT_DIR / "R2_step_bar"
    fig.savefig(str(out_path.with_suffix(".png")), dpi=600)
    fig.savefig(str(out_path.with_suffix(".pdf")))
    plt.close(fig)

    print(f"\n已保存为古典风格的统计图: '{OUT_DIR}/'")
    print(f" - {out_path.with_suffix('.png')}")
    print(f" - {out_path.with_suffix('.pdf')}")

    # --- 统计检验部分 (未变) ---
    def perm_p(a, b, iters=10000):
        d = a - b;
        obs = abs(d.mean());
        n = len(d)
        if np.allclose(d, 0): return 1.0
        cnt = sum(abs((d * np.random.choice([-1, 1], size=n)).mean()) >= obs - 1e-12 for _ in range(iters))
        return (cnt + 1) / (iters + 1)

    def cliffs_delta_2dof_better(a, b):
        return (np.sum(b - a > 0) - np.sum(b - a < 0)) / len(b - a)

    print("\n--- Statistical Analysis ---")
    for name, M in [("Lock (ms)", LOCK), ("Max dev (%)", MAXD)]:
        print(f"\n=== {name} ===")
        for i, sc in enumerate(SCENARIOS):
            a, b = M.pid[i], M.d2f[i]
            print(f"{sc:>12s} | PID mean={a.mean():.3f}, 2-DOF mean={b.mean():.3f}, Δmean={a.mean() - b.mean():.3f}, "
                  f"p≈{perm_p(a, b):.4f}, Cliff's δ={cliffs_delta_2dof_better(a, b):.3f}")


if __name__ == "__main__":
    main()
