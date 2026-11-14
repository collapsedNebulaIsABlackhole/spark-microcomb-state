"""把 Figshare 的实测梳齿表收成一张特征表，并做 K-means。

这些 xlsx 没有孤子数标签。聚类只按光谱形状分组，不把簇名写成单孤子或多孤子。
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "figshare-30026368"
OUT = ROOT / "results"
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def one_file(path: Path, split: str) -> dict:
    frame = pd.read_excel(path)
    frame = frame.dropna(subset=["mu", "power"]).sort_values("mu")
    power = frame["power"].to_numpy(dtype=float)
    mu = frame["mu"].to_numpy(dtype=float)
    peak = float(np.max(power))
    keep = power > peak - 20
    linear = 10 ** (power / 10)
    weight = linear / np.sum(linear)
    return {
        "split": split,
        "file": path.name,
        "n_lines": int(len(frame)),
        "lines_within_20db": int(np.sum(keep)),
        "mu_span_20db": float(mu[keep].max() - mu[keep].min()) if np.any(keep) else 0.0,
        "peak_power_dbm": peak,
        "power_std_db": float(np.std(power)),
        "mu_centroid": float(np.sum(weight * mu)),
    }


def main():
    rows = []
    for split in ("TrainingSet", "ValidationSet", "TestingSet"):
        folder = DATA / split
        files = sorted(folder.glob("*.xlsx"))
        print(split, len(files), flush=True)
        for index, path in enumerate(files, start=1):
            rows.append(one_file(path, split))
            if index % 100 == 0:
                print(f"  {index}", flush=True)
    frame = pd.DataFrame(rows)
    features = ["lines_within_20db", "mu_span_20db", "peak_power_dbm", "power_std_db"]
    scaled = StandardScaler().fit_transform(frame[features])
    frame["cluster"] = KMeans(n_clusters=3, n_init=10, random_state=7).fit_predict(scaled)
    frame.to_csv(OUT / "figshare_features.csv", index=False, encoding="utf-8-sig")
    counts = frame.groupby(["split", "cluster"]).size().reset_index(name="n")
    counts.to_csv(OUT / "figshare_cluster_counts.csv", index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    for cluster, part in frame.groupby("cluster"):
        ax.scatter(part["mu_span_20db"], part["power_std_db"], s=16, alpha=0.75, label=f"簇 {cluster}")
    ax.set_xlabel("20 dB 内的模式跨度")
    ax.set_ylabel("梳齿功率标准差 / dB")
    ax.set_title("实测光谱的三种形状（K-means，没有使用孤子数标签）")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "figshare_clusters.png", dpi=160)
    plt.close(fig)
    print(frame.groupby("cluster")[features].mean())
    print("rows", len(frame))


if __name__ == "__main__":
    main()
