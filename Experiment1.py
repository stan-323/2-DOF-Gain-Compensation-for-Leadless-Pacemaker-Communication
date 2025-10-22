import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter
from enum import Enum

# --- 1. 参数设定 ---

class OscState(Enum):
    QUENCHED = 0
    GROWING = 1

# 信号与时间参数
bit_rate = 1e3
bit_duration = 1 / bit_rate
original_bits = np.array([0, 1, 0, 1, 0, 1, 1])
duration = len(original_bits) * bit_duration
sampling_rate = 50e6
t = np.arange(0, duration, 1/sampling_rate)
dt = 1/sampling_rate

# 物理模型参数 (使用我们最后调试好的版本)
V_th_quench = 1.5
V_th_restart = 0.5
base_growth_factor = 2.5e4
signal_influence = 8e4  # 使用较高的值以使'1'的脉冲更密集
charge_factor = 3.0
tau_quench_discharge = 80e-6
ook_input_strength_1 = 0.5
ook_input_strength_0 = 0.1

# --- 2. 生成基带信号 (不变) ---
baseband_strength = np.zeros_like(t)
for i, bit in enumerate(original_bits):
    start_index = int(i * bit_duration * sampling_rate)
    end_index = int((i + 1) * bit_duration * sampling_rate)
    strength = ook_input_strength_1 if bit == 1 else ook_input_strength_0
    baseband_strength[start_index:end_index] = strength

# --- 3. 仿真循环 (不变) ---
osc_envelope = np.zeros_like(t)
v_quench_bias = np.zeros_like(t)
state = OscState.QUENCHED

for i in range(1, len(t)):
    v_quench_bias[i] = v_quench_bias[i-1]
    osc_envelope[i] = osc_envelope[i-1]

    if state == OscState.QUENCHED:
        osc_envelope[i] = 0.0
        discharge = v_quench_bias[i-1] / tau_quench_discharge
        v_quench_bias[i] -= discharge * dt
        if v_quench_bias[i] < V_th_restart:
            state = OscState.GROWING
            osc_envelope[i] = 1e-6

    elif state == OscState.GROWING:
        effective_growth = base_growth_factor + signal_influence * baseband_strength[i]
        growth = effective_growth * osc_envelope[i-1]
        osc_envelope[i] += growth * dt
        if osc_envelope[i] >= V_th_quench:
            osc_envelope[i] = V_th_quench
            state = OscState.QUENCHED
            v_quench_bias[i] = V_th_quench
        else:
            charge = charge_factor * (osc_envelope[i]**2)
            discharge = v_quench_bias[i-1] / tau_quench_discharge
            v_quench_bias[i] += (charge - discharge) * dt
    if v_quench_bias[i] < 0:
        v_quench_bias[i] = 0

# --- 4. 解调与可视化 (核心修改在此) ---
f_rf_visual = 250e3

# **核心修改 1: 引入直流偏置电压**
V_dc_bias = 2.0  # V

# **核心修改 2: 生成包含直流偏置的、始终为正的振荡信号**
oscillator_signal_visual = V_dc_bias + osc_envelope * np.cos(2 * np.pi * f_rf_visual * t)

# 解调过程不变，因为它只依赖于包络 `osc_envelope`
def lowpass_filter(data, cutoff_freq, fs, order=4):
    b, a = butter(order, cutoff_freq / (fs / 2.0), btype='low', analog=False)
    return lfilter(b, a, data)
demod_analog_output = lowpass_filter(osc_envelope, bit_rate * 2.5, sampling_rate)

# --- 5. 可视化 (更新图2的标题和范围) ---
fig, axs = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
fig.suptitle('Physically Accurate SRO Waveform (with DC Bias)', fontsize=16)

# Plot 1: Input Signal (不变)
axs[0].plot(t, baseband_strength, 'k', linewidth=1.5)
axs[0].set_title("Plot 1: Input Signal Strength")
axs[0].set_ylabel('Equivalent Strength')
axs[0].grid(True, which='both', linestyle='--', linewidth=0.5)

# Plot 2: Oscillator Output (更新)
axs[1].plot(t, oscillator_signal_visual, 'k', linewidth=0.7)
axs[1].set_title("Plot 2: Oscillator's Output (Physically Accurate, No Negative Voltage)")
axs[1].set_ylabel('Voltage (V)')
axs[1].grid(True, which='both', linestyle='--', linewidth=0.5)
# 更新Y轴范围以正确显示波形
axs[1].set_ylim(0, V_dc_bias + V_th_quench + 0.5) # Y轴从0开始

# Plot 3: Demodulated Output (不变)
axs[2].plot(t, demod_analog_output, 'b', linewidth=2)
axs[2].set_title("Plot 3: Final Demodulated Output (from Envelope)")
axs[2].set_xlabel('Time (s)')
axs[2].set_ylabel('Voltage (V)')
axs[2].grid(True, which='both', linestyle='--', linewidth=0.5)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.xlim(0, duration)
plt.show()