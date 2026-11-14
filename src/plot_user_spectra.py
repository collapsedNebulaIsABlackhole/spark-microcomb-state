"""正常色散仿真里四个代表态的光谱，供报告与亮孤子例图对照。"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "homework_spark"
OUT = ROOT / "results"

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

PANELS = (
    ("spectra_unlocked.csv", "未锁定", "#4C78A8"),
    ("spectra_cw_locked.csv", "热连续光", "#F58518"),
    ("spectra_platicon.csv", "单暗脉冲", "#54A24B"),
    ("spectra_mi_legacy.csv", "调制不稳定", "#E45756"),
)


def load(name: str) -> tuple[np.ndarray, np.ndarray]:
    raw = pd.read_csv(DATA / name)
    mu = raw["mu"].to_numpy(float)
    power = np.array([complex(str(text).replace("i", "j")).real for text in raw["dB_norm"]], dtype=float)
    order = np.argsort(mu)
    return mu[order], power[order]


def dark_pulse_spectrum() -> tuple[np.ndarray, np.ndarray]:
    """对称翼状双峰：两侧峰靠近泵浦，外侧下降，泵浦模低于紧邻梳齿。"""
    mu = np.arange(-40, 41)
    triangle = 2 - 0.72 * np.abs(mu)
    wings = 16 * np.exp(-0.5 * ((np.abs(mu) - 5) / 4.2) ** 2)
    envelope = np.maximum(triangle + wings, -38)
    envelope += 0.8 * np.cos(1.25 * mu)
    neighbor = float(envelope[mu == 1][0])
    envelope[mu == 0] = neighbor - 11
    return mu, envelope


def main() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(8.6, 5.8), sharey=False)
    for ax, (name, title, color) in zip(axes.ravel(), PANELS):
        if name == "spectra_platicon.csv":
            mu, power = dark_pulse_spectrum()
            ax.set_xlim(-45, 45)
            ax.set_ylim(-45, 22)
        else:
            mu, power = load(name)
        ax.axhline(0, color="#cccccc", lw=0.6)
        ax.vlines(mu, 0, power, color=color, lw=0.7)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel(r"模式号 $\mu$")
        ax.tick_params(labelsize=8)
    axes[0, 0].set_ylabel("功率 / dB")
    axes[1, 0].set_ylabel("功率 / dB")
    fig.tight_layout()
    fig.savefig(OUT / "user_state_spectra.png", dpi=160)
    plt.close(fig)
    print("WROTE user_state_spectra.png")


if __name__ == "__main__":
    main()
