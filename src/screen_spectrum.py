"""用标准暗孤子光谱筛选实验光谱，并给出仿真库中的参数范围。

标准谱是图 12 第四张，AMX = 0。规则只读光谱；时域类别留在仿真库里作核对。
上传文件为两列：mu, power，功率单位 dBm。
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
TEMPLATE_MIN = 0.35


def wing_band(mu: np.ndarray, ref: np.ndarray) -> np.ndarray:
    return (np.abs(mu) >= 2) & (np.abs(mu) <= 45) & (ref > np.max(ref) - 40)


def template_corr(spec: np.ndarray, ref: np.ndarray, mu: np.ndarray) -> float:
    band = wing_band(mu, ref)
    if np.count_nonzero(band) < 8 or float(np.std(spec[band])) < 1e-6:
        return float("nan")
    return float(np.corrcoef(spec[band], ref[band])[0, 1])


def load():
    table = pd.read_csv(RESULTS / "dark_screen_features.csv")
    pack = np.load(RESULTS / "dark_library.npz")
    return table, pack["spec"], pack["mu"], pack["ref"]


def nearest_interval(query: np.ndarray, table: pd.DataFrame, spec: np.ndarray, mu: np.ndarray, ref: np.ndarray, skip: int | None = None):
    band = wing_band(mu, ref)
    pool = table.index[(table["dark"]) & (table["template_corr"] >= TEMPLATE_MIN)].to_numpy()
    if skip is not None:
        pool = pool[pool != skip]
    scores = []
    for index in pool:
        left = spec[index, band]
        if float(np.std(left)) < 1e-6 or float(np.std(query[band])) < 1e-6:
            scores.append(-np.inf)
        else:
            scores.append(float(np.corrcoef(query[band], left)[0, 1]))
    order = np.argsort(scores)[-5:]
    chosen = table.loc[pool[order]]
    return chosen


def screen(spec: np.ndarray, table: pd.DataFrame, library: np.ndarray, mu: np.ndarray, ref: np.ndarray, skip: int | None = None) -> dict:
    corr = template_corr(spec, ref, mu)
    passed = bool(np.isfinite(corr) and corr >= TEMPLATE_MIN)
    result = {"template_corr": corr, "passed": passed}
    if not passed:
        return result
    chosen = nearest_interval(spec, table, library, mu, ref, skip=skip)
    result["neighbors"] = chosen
    result["delw"] = (float(chosen["delw"].min()), float(chosen["delw"].max()))
    result["P_pump"] = (float(chosen["P_pump"].min()), float(chosen["P_pump"].max()))
    result["AMX_MHz"] = (float(chosen["AMX_MHz"].min()), float(chosen["AMX_MHz"].max()))
    return result


def loo(table: pd.DataFrame, spec: np.ndarray, mu: np.ndarray, ref: np.ndarray) -> None:
    cover = 0
    widths_d, widths_p = [], []
    checked = table.index[table["dark"] & (table["template_corr"] >= TEMPLATE_MIN)]
    for index in checked:
        result = screen(spec[index], table, spec, mu, ref, skip=int(index))
        row = table.loc[index]
        d0, d1 = result["delw"]
        p0, p1 = result["P_pump"]
        cover += int(d0 <= row["delw"] <= d1 and p0 <= row["P_pump"] <= p1)
        widths_d.append(d1 - d0)
        widths_p.append(p1 - p0)
    print(
        f"LOO_NN {cover}/{len(checked)} "
        f"median_delw_width {np.median(widths_d):.3f} median_P_width {np.median(widths_p):.2f}"
    )


def draw(table: pd.DataFrame) -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    grid = table[
        np.isclose(table["AMX_MHz"], 200)
        & np.isclose(table["p_cross"], 1)
        & np.isclose(table["loss_dBcm"], 0.04)
        & np.isclose(table["seta"], 0.011)
        & np.isclose(table["amx_sign"], 1)
        & (table["beta2_ps2_per_km"] > 0)
    ]
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.0))
    ax = axes[0]
    for flag, color, label in ((False, "#E45756", "非暗孤子"), (True, "#1f4e79", "暗孤子")):
        part = table.loc[table["dark"] == flag, "template_corr"]
        ax.scatter(np.full(len(part), 0 if flag else 1), part, c=color, s=28, label=label, zorder=2)
    ax.axhline(TEMPLATE_MIN, color="#333333", lw=0.8, ls="--")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["暗孤子", "非暗孤子"])
    ax.set_ylabel("与标准谱的翼部相关")
    ax.set_xlim(-0.6, 1.6)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    passed = grid["dark"] & (grid["template_corr"] >= TEMPLATE_MIN)
    ax.scatter(grid.loc[~grid["dark"], "delw"], grid.loc[~grid["dark"], "P_pump"], c="#E45756", marker="X", s=36, label="非暗孤子")
    ax.scatter(grid.loc[grid["dark"] & ~passed, "delw"], grid.loc[grid["dark"] & ~passed, "P_pump"], c="#1f4e79", marker="o", s=36, label="暗孤子，形貌偏离")
    ax.scatter(grid.loc[passed, "delw"], grid.loc[passed, "P_pump"], c="#1f4e79", marker="o", s=36)
    ax.scatter(grid.loc[passed, "delw"], grid.loc[passed, "P_pump"], facecolors="none", edgecolors="#54A24B", s=130, linewidths=1.3, label="通过模板且时域确认")
    ax.set_xlabel(r"归一化失谐 $\delta$")
    ax.set_ylabel("泵浦功率 / W")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS / "dark_param_interval.png", dpi=160)
    plt.close(fig)


def main() -> None:
    table, spec, mu, ref = load()
    loo(table, spec, mu, ref)
    draw(table)
    if len(sys.argv) == 1:
        demo = int(table.index[(table["AMX_MHz"] == 0) & np.isclose(table["P_pump"], 0.2)][0])
        result = screen(spec[demo], table, spec, mu, ref, skip=demo)
        print(
            "DEMO standard",
            round(result["template_corr"], 3),
            "delw",
            tuple(round(v, 3) for v in result["delw"]),
            "P",
            tuple(round(v, 2) for v in result["P_pump"]),
            "AMX_MHz",
            tuple(round(v, 1) for v in result["AMX_MHz"]),
        )
        return
    raw = pd.read_csv(sys.argv[1])
    columns = {name.lower(): name for name in raw.columns}
    mode = raw[columns["mu"]].to_numpy(float)
    power = raw[columns["power"]].to_numpy(float)
    full = np.full(mu.shape, np.min(power) - 40)
    for value_mu, value_p in zip(mode, power):
        full[int(np.argmin(np.abs(mu - value_mu)))] = value_p
    result = screen(full, table, spec, mu, ref)
    print(f"模板相关 {result['template_corr']:.3f}")
    if not result["passed"]:
        print("未通过。与标准暗孤子光谱相差较远，不给出参数区间。")
        return
    d0, d1 = result["delw"]
    p0, p1 = result["P_pump"]
    a0, a1 = result["AMX_MHz"]
    print(f"通过。近邻仿真的失谐 {d0:.3f}–{d1:.3f}，泵浦 {p0:.2f}–{p1:.2f} W，AMX {a0:.0f}–{a1:.0f} MHz。")


if __name__ == "__main__":
    main()
