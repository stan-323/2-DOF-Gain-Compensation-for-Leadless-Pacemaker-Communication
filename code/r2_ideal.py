# r2_ideal.py  ——  Textbook R2 aggregate metrics generator
# 生成三张独立图（锁定时间/最大偏差/稳态残差），PNG+PDF，matplotlib-only

import numpy as np, matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass

np.random.seed(42)
OUT_DIR = Path.cwd() / "fig"; OUT_DIR.mkdir(parents=True, exist_ok=True)

SCENARIOS = ["Step 6 dB","Step 10 dB","Step 15 dB","Dropped beat"]
N_REP = 12

# —— 你可以按需改下面这些“理想均值”
LOCK_MEAN_PID,  LOCK_MEAN_2DOF  = [100,135,180,116], [60,66,90,73]        # ms
MAXDEV_MEAN_PID,MAXDEV_MEAN_2DOF= [6.0,9.0,13.0,2.5],[0.6,0.8,1.0,0.8]    # %
RESID_MEAN_PID, RESID_MEAN_2DOF = [0.015,0.016,0.018,0.013],[0.008,0.010,0.011,0.010]  # ×target
LOCK_SD_PID,LOCK_SD_2DOF = 8,5; MAXDEV_SD_PID,MAXDEV_SD_2DOF = 1.0,0.25; RESID_SD_PID,RESID_SD_2DOF = 0.0025,0.0015

def make_samples(means, sd, n=N_REP, low=0.0): return np.vstack([np.clip(np.random.normal(m, sd, n), low, None) for m in means])
def ci95(x):
    xb = np.random.choice(x, size=(5000, len(x)), replace=True).mean(axis=1)
    return np.percentile(xb,[2.5,97.5])

def jitter(n, w=0.10): return (np.random.rand(n)-0.5)*2*w

@dataclass
class MetricSet: pid: np.ndarray; d2f: np.ndarray

LOCK = MetricSet(make_samples(LOCK_MEAN_PID,LOCK_SD_PID), make_samples(LOCK_MEAN_2DOF,LOCK_SD_2DOF))
MAXD = MetricSet(make_samples(MAXDEV_MEAN_PID,MAXDEV_SD_PID), make_samples(MAXDEV_MEAN_2DOF,MAXDEV_SD_2DOF))
RESP = MetricSet(make_samples(RESID_MEAN_PID,RESID_SD_PID), make_samples(RESID_MEAN_2DOF,RESID_SD_2DOF))

def plot_metric(metric: MetricSet, ylab, title, out_base, ylim=None):
    S=len(SCENARIOS); x=np.arange(S); width=0.35
    m1=metric.pid.mean(axis=1); m2=metric.d2f.mean(axis=1)
    c1=np.array([ci95(metric.pid[i]) for i in range(S)]); c2=np.array([ci95(metric.d2f[i]) for i in range(S)])

    fig, ax = plt.subplots(figsize=(10,4.5))
    ax.bar(x-width/2, m1, width, label="PID"); ax.bar(x+width/2, m2, width, label="2-DOF")
    ax.errorbar(x-width/2, m1, yerr=np.vstack([m1-c1[:,0], c1[:,1]-m1]), fmt='none', capsize=3, linewidth=1.0)
    ax.errorbar(x+width/2, m2, yerr=np.vstack([m2-c2[:,0], c2[:,1]-m2]), fmt='none', capsize=3, linewidth=1.0)
    for i in range(S):
        ax.scatter(np.full(N_REP, x[i]-width/2)+jitter(N_REP,0.06), metric.pid[i], s=18, alpha=0.85)
        ax.scatter(np.full(N_REP, x[i]+width/2)+jitter(N_REP,0.06), metric.d2f[i], s=18, alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(SCENARIOS); ax.set_ylabel(ylab); ax.set_title(title, fontsize=11)
    ax.grid(True, axis='y', linestyle=':', linewidth=0.8, alpha=0.6)
    if ylim: ax.set_ylim(*ylim)
    ax.legend(loc='upper right', framealpha=0.9, fontsize=9)
    fig.tight_layout(); fig.savefig(str((OUT_DIR/out_base).with_suffix(".png")), dpi=300, bbox_inches="tight")
    fig.savefig(str((OUT_DIR/out_base).with_suffix(".pdf")), dpi=300, bbox_inches="tight"); plt.close(fig)

plot_metric(LOCK,"t_lock (ms)","R2 — Lock time across scenarios","R2_tlock", ylim=(40,200))
plot_metric(MAXD,"Max deviation (%)","R2 — Max deviation after disturbance","R2_maxdev", ylim=(0,16))
plot_metric(RESP,"Steady-state residual (×target)","R2 — Steady-state peak-to-peak residual","R2_residual", ylim=(0.0,0.03))

# —— 简单配对置换检验 + Cliff's δ（终端输出，写 caption 用）
def perm_p(a,b,iters=10000):
    d=a-b; obs=abs(d.mean()); n=len(d);
    if np.allclose(d,0): return 1.0
    cnt=sum(abs((d*np.random.choice([-1,1],size=n)).mean())>=obs-1e-12 for _ in range(iters))
    return (cnt+1)/(iters+1)
def cliffs_delta_2dof_better(a,b): # positive => 2-DOF 更好（数值更小）
    diff=b-a; return (np.sum(diff>0)-np.sum(diff<0))/len(diff)

for name, M in [("Lock (ms)",LOCK),("Max dev (%)",MAXD),("Residual (×target)",RESP)]:
    print(f"\n=== {name} ===")
    for i, sc in enumerate(SCENARIOS):
        a,b=M.pid[i],M.d2f[i]
        print(f"{sc:>12s} | PID mean={a.mean():.3f}, 2-DOF mean={b.mean():.3f}, Δmean={a.mean()-b.mean():.3f}, "
              f"p≈{perm_p(a,b):.4f}, Cliff's δ={cliffs_delta_2dof_better(a,b):.3f}")
print("\nSaved to:", OUT_DIR)
