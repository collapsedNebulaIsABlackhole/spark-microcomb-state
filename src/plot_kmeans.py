"""K-means 和形态标签的对照图。数字来自虚拟机上的 Spark 作业。"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

shapes = ["窄谱", "平滑包络", "起伏包络"]
cells = np.array(
    [
        [0, 0, 130],
        [0, 156, 0],
        [359, 225, 0],
    ],
    dtype=float,
)
fig, ax = plt.subplots(figsize=(6.4, 4.2))
left = np.zeros(3)
colors = ["#4C78A8", "#54A24B", "#E45756"]
names = ["簇 0", "簇 1", "簇 2"]
for index, (color, name) in enumerate(zip(colors, names)):
    ax.barh(shapes, cells[:, index], left=left, color=color, label=name)
    left += cells[:, index]
ax.set_xlabel("光谱条数")
ax.set_title("Spark K-means 和 sech² 形态的对照")
ax.legend(frameon=False, ncol=3)
fig.tight_layout()
fig.savefig(ROOT / "results" / "public_kmeans_shapes.png", dpi=160)
plt.close(fig)
print("wrote public_kmeans_shapes.png")
