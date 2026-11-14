"""从 Figshare 实测光谱提取特征，并按补充材料的 sech^2 包络吻合度给出形态标签。

标签不是作者公布的孤子数。补充材料写明：单孤子光谱接近 sech^2，多孤子会在包络上产生干涉起伏。
这里用包络拟合残差区分这两种形态，窄谱单独成一类。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "figshare-30026368"
OUT = ROOT / "results" / "public_spectra_features.csv"


def sech_residual(mu: np.ndarray, power_dbm: np.ndarray) -> float:
    keep = np.abs(mu) > 1
    mu = mu[keep]
    linear = 10 ** (power_dbm[keep] / 10)
    if len(mu) < 8:
        return float("nan")
    center = float(np.sum(linear * mu) / np.sum(linear))
    offset = mu - center
    best = 1e9
    for width in np.linspace(4, 70, 34):
        envelope = 1 / np.cosh(offset / width) ** 2
        scale = float(np.dot(envelope, linear) / np.dot(envelope, envelope))
        error = np.sqrt(np.mean((linear - scale * envelope) ** 2)) / (float(np.mean(linear)) + 1e-30)
        if error < best:
            best = error
    return float(best)


def one_file(path: Path, split: str) -> dict:
    frame = pd.read_excel(path).dropna(subset=["mu", "power"]).sort_values("mu")
    mu = frame["mu"].to_numpy(dtype=float)
    power = frame["power"].to_numpy(dtype=float)
    peak = float(np.max(power))
    loud = power > peak - 20
    linear = 10 ** (power / 10)
    weight = linear / np.sum(linear)
    return {
        "split": split,
        "file": path.name,
        "n_lines": int(len(frame)),
        "lines_within_20db": int(np.sum(loud)),
        "mu_span_20db": float(mu[loud].max() - mu[loud].min()) if np.any(loud) else 0.0,
        "peak_power_dbm": peak,
        "mu_centroid": float(np.sum(weight * mu)),
        "sech_rms": sech_residual(mu, power),
    }


def main() -> None:
    rows = []
    for split in ("TrainingSet", "ValidationSet", "TestingSet"):
        files = sorted((DATA / split).glob("*.xlsx"))
        print(split, len(files), flush=True)
        for index, path in enumerate(files, start=1):
            rows.append(one_file(path, split))
            if index % 100 == 0:
                print(f"  {index}", flush=True)
    frame = pd.DataFrame(rows)
    OUT.parent.mkdir(exist_ok=True)
    frame.to_csv(OUT, index=False, encoding="utf-8-sig")
    wide = frame[frame["mu_span_20db"] >= 50]
    print("rows", len(frame))
    print(frame.groupby("split").size())
    print("sech_rms quantiles", wide["sech_rms"].quantile([0.1, 0.25, 0.5, 0.75, 0.9]).to_dict())
    print("span quantiles", frame["mu_span_20db"].quantile([0.1, 0.25, 0.5, 0.75, 0.9]).to_dict())


if __name__ == "__main__":
    main()
