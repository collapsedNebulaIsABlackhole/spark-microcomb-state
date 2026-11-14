"""扫描泵浦和失谐，对比“带着场往红失谐扫”和“在目标失谐从噪声冷启动”。

冷启动停在连续光或多峰抖动，扫描才能进入单孤子。这是本作业用来解释
“仿真一直停在调制不稳定性”的数值实验。
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lle import label_state, propagate, spectrum_db

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

STATE_CN = {
    "continuous_wave": "连续光",
    "modulation_instability": "调制不稳定性",
    "chaotic_or_breathing": "混沌或呼吸",
    "single_soliton": "单孤子",
    "multi_soliton": "多孤子",
}
STATE_COLOR = {
    "continuous_wave": "#4C78A8",
    "modulation_instability": "#F58518",
    "chaotic_or_breathing": "#E45756",
    "single_soliton": "#54A24B",
    "multi_soliton": "#B279A2",
}


def integrate_scan(pump, alphas, d2=1.0, n=256, steps=2200, seed=2):
    psi = None
    rows = []
    for alpha in alphas:
        psi, energy, _ = propagate(
            float(alpha), pump, d2=d2, n=n, dt=0.01, steps=steps, seed=seed, psi0=psi
        )
        info = label_state(psi, energy)
        info.update(alpha=float(alpha), pump=float(pump), d2=float(d2), protocol="scan")
        info["spectrum"] = spectrum_db(psi)
        info["intensity"] = np.abs(psi) ** 2
        rows.append(info)
    return rows


def integrate_cold(pump, alphas, d2=1.0, n=256, steps=4000, seed=2):
    rows = []
    for alpha in alphas:
        psi, energy, _ = propagate(
            float(alpha), pump, d2=d2, n=n, dt=0.01, steps=steps, seed=seed, psi0=None
        )
        info = label_state(psi, energy)
        info.update(alpha=float(alpha), pump=float(pump), d2=float(d2), protocol="cold_start")
        info["spectrum"] = spectrum_db(psi)
        info["intensity"] = np.abs(psi) ** 2
        rows.append(info)
    return rows


def write_table(rows, path):
    fields = [
        "protocol", "pump", "alpha", "d2", "state", "peaks",
        "lines_above_20db", "flicker", "mean_power", "peak_power",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fields})


def plot_scan(rows, path):
    pumps = sorted({row["pump"] for row in rows})
    fig, axes = plt.subplots(len(pumps), 1, figsize=(8.2, 2.15 * len(pumps)), sharex=True)
    if len(pumps) == 1:
        axes = [axes]
    for ax, pump in zip(axes, pumps):
        subset = [row for row in rows if row["pump"] == pump]
        ax.plot([row["alpha"] for row in subset], [row["mean_power"] for row in subset], color="#333333", lw=1.2)
        for row in subset:
            ax.scatter(row["alpha"], row["mean_power"], s=36, color=STATE_COLOR[row["state"]], zorder=3)
        ax.set_ylabel("平均腔内功率")
        ax.set_title(f"扫描协议，泵浦幅度 F = {pump:g}，异常色散")
    axes[-1].set_xlabel("归一化失谐 α（单位：场衰减率）")
    handles = [
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=color, label=STATE_CN[name], markersize=8)
        for name, color in STATE_COLOR.items()
    ]
    axes[0].legend(handles=handles, ncol=3, fontsize=8, frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_examples(scan_rows, cold_rows, path):
    soliton = next(row for row in scan_rows if row["state"] == "single_soliton")
    early = min(scan_rows, key=lambda row: abs(row["alpha"] - 2.0))
    cold = min(cold_rows, key=lambda row: abs(row["alpha"] - soliton["alpha"]))
    panels = [
        (early, "扫描停在较小失谐"),
        (soliton, "同一扫描继续进入单孤子"),
        (cold, "在孤子失谐处从噪声冷启动"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(10.2, 5.6))
    for col, (row, title) in enumerate(panels):
        axes[0, col].plot(row["intensity"], color=STATE_COLOR[row["state"]])
        axes[0, col].ticklabel_format(useOffset=False, axis="y")
        axes[0, col].set_ylim(bottom=0)
        axes[0, col].set_title(f"{title}\nα={row['alpha']:.2f}，{STATE_CN[row['state']]}", fontsize=10)
        axes[0, col].set_xlabel("快时间采样点")
        axes[0, col].set_ylabel("光强")
        modes = np.arange(len(row["spectrum"])) - len(row["spectrum"]) // 2
        axes[1, col].plot(modes, row["spectrum"], color="#333333", lw=0.9)
        axes[1, col].set_ylim(-50, 1)
        axes[1, col].set_xlabel("模式号")
        axes[1, col].set_ylabel("相对功率 / dB")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_map(rows, path, title):
    fig, ax = plt.subplots(figsize=(8.4, 3.8))
    for name, color in STATE_COLOR.items():
        picked = [row for row in rows if row["state"] == name]
        if not picked:
            continue
        ax.scatter(
            [row["alpha"] for row in picked],
            [row["pump"] for row in picked],
            s=42,
            color=color,
            label=STATE_CN[name],
        )
    ax.set_xlabel("归一化失谐 α")
    ax.set_ylabel("泵浦幅度 F")
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    alphas = np.linspace(0.4, 12.0, 18)
    pumps = [1.6, 2.4, 3.2, 4.2]
    scan_rows = []
    cold_rows = []
    for pump in pumps:
        print(f"scan F={pump}", flush=True)
        scan_rows.extend(integrate_scan(pump, alphas))
        cold_alphas = [2.0, 7.2, 11.0]
        print(f"cold F={pump}", flush=True)
        cold_rows.extend(integrate_cold(pump, cold_alphas))
    print("normal-dispersion scan", flush=True)
    normal_rows = integrate_scan(3.2, alphas, d2=-1.0, seed=3)
    for row in normal_rows:
        row["protocol"] = "scan_normal_dispersion"

    write_table(scan_rows + cold_rows + normal_rows, OUT / "lle_states.csv")
    plot_scan(scan_rows, OUT / "scan_power_states.png")
    reference = [row for row in scan_rows if row["pump"] == 3.2]
    reference_cold = [row for row in cold_rows if row["pump"] == 3.2]
    plot_examples(reference, reference_cold, OUT / "mi_versus_soliton.png")
    plot_map(scan_rows, OUT / "state_map_scan.png", "异常色散、失谐扫描：最终状态")
    plot_map(
        [row for row in cold_rows],
        OUT / "state_map_cold.png",
        "异常色散、冷启动：同一失谐往往到不了孤子",
    )

    soliton = [row for row in scan_rows if row["state"] == "single_soliton"]
    print("single soliton count", len(soliton))
    if soliton:
        alphas_s = [row["alpha"] for row in soliton]
        print("soliton alpha range", min(alphas_s), max(alphas_s))
    print("normal states", sorted({row["state"] for row in normal_rows}))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
