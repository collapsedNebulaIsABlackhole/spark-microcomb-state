"""第 3 节对标第 2 节的两张分类图：跨度—残差，以及三项形貌量。"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

PUBLIC_COLOR = {"narrow": "#9D755D", "smooth": "#54A24B", "modulated": "#E45756"}
PUBLIC_NAME = {"narrow": "窄谱", "smooth": "平滑包络", "modulated": "起伏包络"}
TIME_ORDER = ("暗孤子", "双坑", "双峰", "起伏")
TIME_COLOR = {"暗孤子": "#1f4e79", "双坑": "#F58518", "双峰": "#B279A2", "起伏": "#333333"}
TIME_MARKER = {"暗孤子": "o", "双坑": "^", "双峰": "s", "起伏": "X"}


def time_label(kind: str) -> str:
    return {"多坑": "双坑", "多峰": "双峰"}.get(kind, kind)


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    tooth = pd.read_csv(RESULTS / "course_method_rank.csv")
    states = pd.read_csv(RESULTS / "public_spectra_states.csv")
    public = tooth.merge(
        states[["split", "file", "sech_rms", "mu_span_20db"]],
        on=["split", "file"],
        how="left",
    )
    middle = pd.read_csv(RESULTS / "middle_transfer_features.csv")
    middle["time_label"] = middle["time_kind"].map(time_label)
    return public, middle


def span_figure(public: pd.DataFrame, middle: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    for shape in ("narrow", "smooth", "modulated"):
        part = public[public["shape"] == shape]
        ax.scatter(
            part["mu_span_20db"],
            part["sech_rms"],
            s=12,
            alpha=0.45,
            c=PUBLIC_COLOR[shape],
            label=PUBLIC_NAME[shape],
            linewidths=0,
            zorder=1,
        )
    ax.axvline(50, color="#333333", lw=0.8, ls="--", zorder=2)
    ax.axhline(0.5, color="#333333", lw=0.8, ls="--", zorder=2)
    for label in TIME_ORDER:
        part = middle[(middle["time_label"] == label) & middle["sech_rms"].notna()]
        ax.scatter(
            part["mu_span_20db"],
            part["sech_rms"],
            s=42,
            c=TIME_COLOR[label],
            marker=TIME_MARKER[label],
            edgecolors="white",
            linewidths=0.6,
            label=label,
            zorder=3,
        )
    ax.set_xlabel("20 dB 模式跨度")
    ax.set_ylabel(r"相对 sech$^{2}$ 包络的残差")
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(RESULTS / "middle_span_rms.png", dpi=160)
    plt.close(fig)


def shape_figure(public: pd.DataFrame, middle: pd.DataFrame) -> None:
    panels = (
        ("adj_db", "相邻梳齿功率差 / dB"),
        ("sech_rms", r"相对 sech$^{2}$ 的残差"),
        ("sym_mae", "左右对称功率差 / dB"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(9.8, 3.7))
    for ax, (feature, ylabel) in zip(axes, panels):
        data = [public.loc[public["shape"] == shape, feature].dropna().to_numpy() for shape in ("narrow", "smooth", "modulated")]
        data.append(middle[feature].dropna().to_numpy())
        box = ax.boxplot(
            data,
            tick_labels=["窄谱", "平滑包络", "起伏包络", "九条光谱"],
            showfliers=False,
            patch_artist=True,
            widths=0.62,
        )
        for patch, color in zip(box["boxes"], ("#9D755D", "#54A24B", "#E45756", "#d9d9d9")):
            patch.set_facecolor(color)
            patch.set_alpha(0.8)
        usable = middle.dropna(subset=[feature])
        for label in TIME_ORDER:
            part = usable[usable["time_label"] == label]
            if part.empty:
                continue
            xs = 4 + np.linspace(-0.12, 0.12, len(part))
            ax.scatter(
                xs,
                part[feature],
                s=28,
                c=TIME_COLOR[label],
                marker=TIME_MARKER[label],
                edgecolors="white",
                linewidths=0.4,
                zorder=3,
                label=label,
            )
        ax.set_ylabel(ylabel)
        ax.tick_params(labelsize=8)
    handles, labels = axes[0].get_legend_handles_labels()
    axes[2].legend(handles, labels, frameon=False, fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(RESULTS / "middle_shape_compare.png", dpi=160)
    plt.close(fig)


def main() -> None:
    public, middle = load()
    span_figure(public, middle)
    shape_figure(public, middle)
    print("WROTE middle_span_rms.png middle_shape_compare.png")


if __name__ == "__main__":
    main()
