"""归一化 Lugiato-Lefever 方程的分步傅里叶求解。

采用的方程（异常色散时 D2 > 0，亮孤子在红失谐一侧）：

    dψ/dt = -(1 + i α) ψ + i |ψ|^2 ψ + i (D2/2) ∂²ψ/∂θ² + F

α 是泵浦相对腔共振的失谐，F 是泵浦幅度。光子寿命已归一成 1。
"""

from __future__ import annotations

import numpy as np


def propagate(alpha, pump, d2=1.0, n=512, dt=0.01, steps=4000, seed=0, psi0=None):
    """从噪声或给定初值积分到 steps 步，返回末态复场、时间轨迹的能量。"""
    theta = np.linspace(-np.pi, np.pi, n, endpoint=False)
    k = np.fft.fftfreq(n, d=theta[1] - theta[0]) * 2 * np.pi
    linear = np.exp((-1 - 1j * alpha - 1j * (d2 / 2) * (k ** 2)) * dt)
    rng = np.random.default_rng(seed)
    if psi0 is None:
        psi = 0.01 * (rng.normal(size=n) + 1j * rng.normal(size=n))
    else:
        psi = np.array(psi0, dtype=np.complex128)
    energy = np.empty(steps)
    for i in range(steps):
        psi = psi * np.exp(1j * np.abs(psi) ** 2 * dt)
        psi = np.fft.ifft(np.fft.fft(psi) * linear)
        psi = psi + pump * dt
        energy[i] = np.mean(np.abs(psi) ** 2)
    return psi, energy, theta


def spectrum_db(psi):
    power = np.abs(np.fft.fftshift(np.fft.fft(psi))) ** 2
    power = power / np.max(power)
    return 10 * np.log10(power + 1e-12)


def count_peaks(intensity, rel=0.45, min_sep=8, floor=1.0):
    """只统计明显高于噪声、并且绝对功率过了 floor 的峰。失谐太大时腔内几乎没光，不能把数值噪声当成调制不稳定性。"""
    peak = float(np.max(intensity))
    if peak < floor:
        return 0
    y = intensity / (peak + 1e-12)
    peaks = []
    for i in range(len(y)):
        if y[i] >= rel and y[i] >= y[i - 1] and y[i] > y[(i + 1) % len(y)]:
            if peaks:
                dists = [min((i - p) % len(y), (p - i) % len(y)) for p in peaks]
                nearest = int(np.argmin(dists))
                if dists[nearest] < min_sep:
                    if y[i] > y[peaks[nearest]]:
                        peaks[nearest] = i
                    continue
            peaks.append(i)
    return len(peaks)


def label_state(psi, energy):
    """用时域峰数和最后一段能量起伏给状态起名。规则写在报告里，便于答辩时解释。"""
    intensity = np.abs(psi) ** 2
    tail = energy[int(0.8 * len(energy)):]
    flicker = np.std(tail) / (np.mean(tail) + 1e-12)
    spec = spectrum_db(psi)
    loud = np.sum(spec > -20)
    peaks = count_peaks(intensity)
    if np.max(intensity) < 1.0 and loud <= 5:
        name = "continuous_wave"
        return {
            "state": name,
            "peaks": int(peaks),
            "lines_above_20db": int(loud),
            "flicker": float(flicker),
            "mean_power": float(np.mean(intensity)),
            "peak_power": float(np.max(intensity)),
        }
    if flicker > 0.08:
        name = "chaotic_or_breathing"
    elif loud <= 5 and peaks <= 1 and np.max(intensity) < 1.5:
        name = "continuous_wave"
    elif peaks <= 1 and loud >= 12:
        name = "single_soliton"
    elif 2 <= peaks <= 6:
        name = "multi_soliton" if flicker <= 0.08 else "chaotic_or_breathing"
    elif peaks > 6:
        name = "modulation_instability"
    else:
        name = "modulation_instability" if loud > 5 else "continuous_wave"
    return {
        "state": name,
        "peaks": int(peaks),
        "lines_above_20db": int(loud),
        "flicker": float(flicker),
        "mean_power": float(np.mean(intensity)),
        "peak_power": float(np.max(intensity)),
    }
