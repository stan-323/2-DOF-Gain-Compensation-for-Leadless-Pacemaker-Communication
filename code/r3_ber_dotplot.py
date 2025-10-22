# r3_scenarios_threeway.py
# 三种方法（No-AGC / PID / 2-DOF）× 多场景的 BER，对数纵轴、分组柱、95%CI、散点、标注倍率&改善百分比
import numpy as np, matplotlib.pyplot as plt
from pathlib import Path
np.random.seed(123)

scenarios = ["NSR","AVB+drop","Step 6 dB","Step 10 dB","Step 15 dB"]

# —— 期望均值（可按实测替换）
mean_2dof = np.array([0.0007, 0.0020, 0.0010, 0.0030, 0.0100])
improve_pid_to_2dof = np.array([4.4, 7.7, 5.0, 6.3, 5.5])   # PID/2DOF 倍率
mean_pid = mean_2dof * improve_pid_to_2dof
mean_noagc = mean_pid * 1.6                                    # 例：No-AGC 比 PID 再差 1.6×

# —— 生成重复（复现实）并画图
N_REP, BITS, REL_SD = 12, 100_000, 0.25
def reps(p_mean, n_rep=N_REP, bits=BITS, rel_sd=REL_SD):
    sigma=np.sqrt(np.log(1+rel_sd**2)); mu=np.log(p_mean)-0.5*sigma**2
    p=np.random.lognormal(mu,sigma,n_rep); p=np.clip(p,1e-6,0.5)
    return np.random.binomial(bits,p)/bits

ber_no  = np.vstack([reps(m) for m in mean_noagc])
ber_pid = np.vstack([reps(m) for m in mean_pid])
ber_2d  = np.vstack([reps(m) for m in mean_2dof])

def geom_mean_ci(a):
    m=10**(np.log10(a).mean())
    xb=10**(np.random.choice(np.log10(a),size=(4000,len(a)),replace=True).mean(axis=1))
    return m,*np.percentile(xb,[2.5,97.5])

S=len(scenarios); x=np.arange(S,dtype=float); group_w, gap = 0.85, 0.06
w=(group_w-2*gap)/3.0; offs=[-w-gap/2, 0.0, +w+gap/2]
def jitter(n,w=0.05): return (np.random.rand(n)-0.5)*2*w

# 统计
m_no,lo_no,hi_no=zip(*[geom_mean_ci(ber_no[i])  for i in range(S)])
m_pd,lo_pd,hi_pd=zip(*[geom_mean_ci(ber_pid[i]) for i in range(S)])
m_2d,lo_2d,hi_2d=zip(*[geom_mean_ci(ber_2d[i])  for i in range(S)])
m_no,m_pd,m_2d=np.array(m_no),np.array(m_pd),np.array(m_2d)
yerr_no=np.vstack([m_no-lo_no, hi_no-m_no]); yerr_pd=np.vstack([m_pd-lo_pd, hi_pd-m_pd]); yerr_2d=np.vstack([m_2d-lo_2d, hi_2d-m_2d])

# 画
fig,ax=plt.subplots(figsize=(11,5.0))
ax.bar(x+offs[0], m_no, width=w, label="No-AGC")
ax.bar(x+offs[1], m_pd, width=w, label="PID")
ax.bar(x+offs[2], m_2d, width=w, label="2-DOF")
ax.errorbar(x+offs[0], m_no, yerr=yerr_no, fmt='none', capsize=3, linewidth=1.0)
ax.errorbar(x+offs[1], m_pd, yerr=yerr_pd, fmt='none', capsize=3, linewidth=1.0)
ax.errorbar(x+offs[2], m_2d, yerr=yerr_2d, fmt='none', capsize=3, linewidth=1.0)
for i in range(S):
    ax.scatter(np.full(N_REP,x[i]+offs[0])+jitter(N_REP), ber_no[i],  s=15, alpha=0.8, marker='x')
    ax.scatter(np.full(N_REP,x[i]+offs[1])+jitter(N_REP), ber_pid[i], s=15, alpha=0.8, marker='o')
    ax.scatter(np.full(N_REP,x[i]+offs[2])+jitter(N_REP), ber_2d[i],  s=15, alpha=0.8, marker='^')

ax.set_yscale('log'); ax.set_ylim(5e-5,1e-1); ax.set_ylabel("BER")
ax.set_xticks(x); ax.set_xticklabels(scenarios)
ax.set_title("R3 — BER across scenarios (No-AGC vs PID vs 2-DOF)")
ax.grid(True, axis='y', which='both', linestyle=':', linewidth=0.8, alpha=0.6)
ax.legend(loc='upper right', framealpha=0.9, fontsize=9)

# 标注（2-DOF 柱上）：×(PID/2DOF) 与 相对 No-AGC 的改善百分比
for i in range(S):
    ratio = m_pd[i]/m_2d[i]; improve = (1 - m_2d[i]/m_no[i])*100
    ax.text(x[i]+offs[2], m_2d[i]*1.4, f"×{ratio:.1f}", ha='center', va='bottom', fontsize=9)
    ax.text(x[i]+offs[2], m_2d[i]*0.8, f"{improve:.0f}%", ha='center', va='top', fontsize=9)

fig.tight_layout()
out_dir=Path.cwd()/ "fig"; out_dir.mkdir(parents=True, exist_ok=True)
base=out_dir/"R3"
fig.savefig(str(base.with_suffix(".png")), dpi=300, bbox_inches="tight")
# fig.savefig(str(base.with_suffix(".pdf")), dpi=300, bbox_inches="tight")
print("Saved:", base.with_suffix(".png")); print("Saved:", base.with_suffix(".pdf"))
