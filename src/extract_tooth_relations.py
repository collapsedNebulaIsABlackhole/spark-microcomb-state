"""从 Figshare 每根梳齿的功率抽出与绝对功率无关的关系量。

没有相位。这些量描述齿与齿之间的功率是否整齐、是否左右对称、是否隔一齿更强。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "figshare-30026368"
STATES = ROOT / "results" / "public_spectra_states.csv"
OUT = ROOT / "results" / "tooth_relations.csv"


def one_file(path: Path, split: str) -> dict:
    frame = pd.read_excel(path).dropna(subset=["mu", "power", "frequency"]).sort_values("mu")
    mu = frame["mu"].to_numpy(dtype=float)
    power = frame["power"].to_numpy(dtype=float)
    freq = frame["frequency"].to_numpy(dtype=float)
    linear = 10 ** (power / 10)
    dmu = np.diff(mu)
    adjacent = np.abs(np.diff(power))[np.isclose(dmu, 1)]
    left = float(linear[mu < 0].sum())
    right = float(linear[mu > 0].sum())
    even = float(linear[np.round(mu) % 2 == 0].sum())
    odd = float(linear[np.round(mu) % 2 != 0].sum())
    steps = np.abs(np.diff(freq))
    steps = steps[steps > 1e-4]
    base = float(np.median(steps)) if len(steps) else np.nan
    gap = float(np.mean(steps > 1.5 * base)) if len(steps) else np.nan
    pump_at = int(np.argmin(np.abs(mu)))
    others = np.delete(power, pump_at)
    distance = np.abs(mu)
    if np.std(distance) > 0 and np.std(power) > 0:
        slope = float(np.corrcoef(distance, power)[0, 1])
    else:
        slope = float("nan")
    return {
        "split": split,
        "file": path.name,
        "adj_db": float(np.mean(adjacent)) if len(adjacent) else np.nan,
        "left_share": left / (left + right) if left + right else np.nan,
        "even_share": even / (even + odd) if even + odd else np.nan,
        "gap_frac": gap,
        "pump_excess_db": float(power[pump_at] - np.median(others)),
        "envelope_corr": slope,
        "n_teeth": int(len(frame)),
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
    if STATES.exists():
        states = pd.read_csv(STATES)[["split", "file", "shape"]]
        frame = frame.merge(states, on=["split", "file"], how="left")
    OUT.parent.mkdir(exist_ok=True)
    frame.to_csv(OUT, index=False, encoding="utf-8-sig")
    print("rows", len(frame), "shapes", frame["shape"].value_counts().to_dict() if "shape" in frame else None)
    print("TOOTH_DONE", OUT)


if __name__ == "__main__":
    main()
