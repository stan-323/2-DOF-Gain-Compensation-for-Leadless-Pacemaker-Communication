# 2-DOF Gain Compensation for Leadless Pacemaker Communication

![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)
![Python 3.x](https://img.shields.io/badge/Python-3.x-green.svg)
![Status: Under Review](https://img.shields.io/badge/Status-Under_Review-orange.svg)

This repository contains the official data processing, evaluation code, and experimental datasets for the research paper: **"Enhancing Communication Robustness for Leadless Pacemakers: 2-DOF Gain Compensation Across Physiologic and Pathologic Dynamics"** (currently under review for *IEEE Transactions on Biomedical Engineering - TBME*).

## 📖 Overview

Implantable Medical Devices (IMDs), particularly multi-chamber Leadless Pacemakers (LCPs), rely on Galvanic Conductive Communication (GCC). However, the intracardiac channel exhibits severe, beat-synchronous signal fading due to cardiac motion. Under pathological conditions like **Mobitz Type II Atrioventricular Block (AVB)**, these fades become aperiodic, abrupt, and deep, causing conventional feedback-only Automatic Gain Control (AGC) systems to fail.

This project introduces a novel **beat-synchronous, two-degree-of-freedom (2-DOF) Hybrid-AGC** architecture for Super-Regenerative Receivers (SRR). By utilizing the peak Received Signal Strength Indicator (RSSI) as an ECG-free proxy for cardiac rhythm, the system combines predictive feedforward compensation with incremental PID feedback to proactively stabilize the input signal against pathologically-triggered step changes.

## ✨ Key Contributions
* **Pathological Channel Robustness:** Specifically designed to maintain stable communication during severe, unpredictable signal fades (e.g., dropped beats in AV block).
* **2-DOF Active Compensation:** A computationally lightweight strategy integrating EMA-based peak RSSI prediction and feedback error correction.
* **Dual-Path Experimental Validation:** Validated via Hardware-in-the-Loop (HIL) simulations for transient response and a programmable ex-vivo porcine heart platform for end-to-end Bit Error Rate (BER) assessment.
* **Significant Performance Gains:** Stabilizes input signals to within 1% of the target amidst >15 dB variations, yielding up to a 60-fold improvement in BER at 5 kbps.

## 📂 Repository Structure

The repository includes Python scripts for offline data processing and algorithm evaluation based on data captured by high-speed DAQ systems:

* `code/`: Core mathematical models and signal processing modules.
* `Experiment1.py`: Script for evaluating the transient response (settling time, maximum overshoot) of the AGC control algorithm.
* `R5_BER1.py` - `R5_BER3.py`: Scripts for calculating the Bit Error Rate (BER) across various data rates under simulated pathological fading conditions (comparing No-AGC, PID, and 2-DOF strategies).
* `*.csv`: Sample raw waveform datasets captured during ex-vivo and HIL experiments.
* `Figure/`: Directory containing generated plots and result visualizations.

## 🚀 How to Run

1. Ensure you have standard scientific Python libraries installed (`numpy`, `scipy`, `matplotlib`, `pandas`).
2. Clone this repository to your local machine.
3. Run the evaluation scripts directly. For example, to reproduce the transient response evaluation:
   ```bash
   python Experiment1.py
