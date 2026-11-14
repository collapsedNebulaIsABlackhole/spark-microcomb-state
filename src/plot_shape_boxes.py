"""把相邻齿功率差、sech² 残差和对称功率差画成同一张图的三个子图。"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

SHAPES = ("narrow", "smooth", "modulated")
NAMES = {"narrow": "窄谱", "smooth": "平滑包络", "modulated": "起伏包络"}
COLORS = {"narrow": "#4C78A8", "smooth": "#54A24B", "modulated": "#E45756"}
PANELS = (
    ("adj_db", "相邻梳齿功率差 / dB"),
    ("sech_rms", r"相对 sech$^{2}$ 的残差"),
    ("sym_mae", "左右对称功率差 / dB"),
)


def main() -> None:
    tooth = pd.read_csv(RESULTS / "course_method_rank.csv")
    states = pd.read_csv(RESULTS / "public_spectra_states.csv")[["split", "file", "sech_rms"]]
    frame = tooth.merge(states, on=["split", "file"], how="left")
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.55))
    for ax, (feature, label) in zip(axes, PANELS):
        data = [frame.loc[frame["shape"] == shape, feature].dropna().to_numpy() for shape in SHAPES]
        box = ax.boxplot(
            data,
            tick_labels=[NAMES[shape] for shape in SHAPES],
            showfliers=False,
            patch_artist=True,
        )
        for patch, shape in zip(box["boxes"], SHAPES):
            patch.set_facecolor(COLORS[shape])
            patch.set_alpha(0.85)
        ax.set_ylabel(label)
        ax.tick_params(labelsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS / "public_shape_boxes.png", dpi=160)
    plt.close(fig)
    print("WROTE public_shape_boxes.png")


if __name__ == "__main__":
    main()
