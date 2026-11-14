"""用你自己的闭环扫描表说明：单暗脉冲已经出现，缺的是宽度台阶。"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "homework_spark" / "operating_points.csv"
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

STATE_CN = {
    "unlocked": "未锁",
    "cw_locked": "热连续光",
    "platicon": "单暗脉冲",
    "platicon_crystal": "双暗脉冲",
    "mi_legacy": "旧符号下的调制不稳定性",
    "other": "过渡",
}
STATE_COLOR = {
    "unlocked": "#4C78A8",
    "cw_locked": "#F58518",
    "platicon": "#54A24B",
    "platicon_crystal": "#B279A2",
    "mi_legacy": "#E45756",
    "other": "#9D755D",
}


def main():
    frame = pd.read_csv(DATA)
    fig, axes = plt.subplots(2, 1, figsize=(8.6, 6.4), sharex=True)
    for state, color in STATE_COLOR.items():
        part = frame[frame["state"] == state]
        if part.empty:
            continue
        axes[0].scatter(part["alpha_L"], part["Pep"], s=12, c=color, label=STATE_CN[state], alpha=0.85)
    axes[0].set_ylabel("腔内功率 Pep")
    axes[0].set_title("闭环自注入锁定扫描：正常色散，D2/2π = -89.30 kHz")
    axes[0].legend(frameon=False, ncol=3, fontsize=8)
    axes[0].axvline(-58.4, color="#333333", lw=0.8, ls="--")

    platicon = frame[(frame["detuning_mode"] == "rev_memory") & (frame["state"] == "platicon")].sort_values("alpha_L")
    slope, intercept = np.polyfit(platicon["alpha_L"], platicon["dark_frac"], 1)
    fitted = slope * platicon["alpha_L"] + intercept
    residual = platicon["dark_frac"] - fitted
    axes[1].scatter(platicon["alpha_L"], platicon["dark_frac"], s=12, c="#54A24B", label="暗区占比")
    axes[1].plot(platicon["alpha_L"], fitted, color="#333333", lw=1.2, label=f"直线斜率 {slope:.4f} / 单位 α_L")
    axes[1].set_xlabel("归一化失谐 α_L = 2 δω / κ")
    axes[1].set_ylabel("暗区占比 dark_frac")
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "your_platicon_branch.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    ax.axhline(0, color="#bbbbbb", lw=0.8)
    ax.scatter(platicon["alpha_L"], residual, s=12, c="#54A24B")
    ax.set_xlabel("归一化失谐 α_L")
    ax.set_ylabel("暗区占比相对直线的残差")
    ax.set_title(f"没有宽度台阶：残差标准差 {residual.std():.4f}，最大绝对值 {residual.abs().max():.4f}")
    fig.tight_layout()
    fig.savefig(OUT / "your_platicon_residual.png", dpi=160)
    plt.close(fig)

    summary = {
        "rows": int(len(frame)),
        "platicon_rev": int(len(platicon)),
        "n_dark_unique": sorted(platicon["n_dark"].dropna().unique().tolist()),
        "n_dip_median": float(platicon["n_dip"].median()),
        "dark_frac_slope": float(slope),
        "residual_std": float(residual.std()),
        "residual_max_abs": float(residual.abs().max()),
        "d2_over_kappa": -89.30 / (18.60 * 1000),
    }
    print(summary)


if __name__ == "__main__":
    main()
