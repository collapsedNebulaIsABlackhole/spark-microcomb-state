"""比较目标暗孤子与其余光谱，寻找只依赖光谱的复合条件。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MU = np.arange(-128, 128)


def metrics(spec: np.ndarray) -> dict:
    peak = float(np.max(spec))
    teeth = spec > peak - 30
    mu_t = MU[teeth]
    left_reach = float(-mu_t[mu_t < 0].min()) if np.any(mu_t < 0) else 0.0
    right_reach = float(mu_t[mu_t > 0].max()) if np.any(mu_t > 0) else 0.0
    both = min(left_reach, right_reach)
    order = np.argsort(MU[teeth])
    power = spec[teeth][order]
    modes = MU[teeth][order]
    adj = np.abs(np.diff(power))
    adj = adj[np.isclose(np.diff(modes), 1)]
    wing = adj[np.abs(modes[:-1][np.isclose(np.diff(modes), 1)]) >= 3]
    core_modes = modes[:-1][np.isclose(np.diff(modes), 1)]
    core = adj[np.abs(core_modes) <= 1] if len(adj) else np.array([])
    pump = float(spec[np.argmin(np.abs(MU))])
    neighbors = spec[(np.abs(MU) >= 2) & (np.abs(MU) <= 6)]
    pump_excess = pump - float(np.median(neighbors))

    def rough(sign: int) -> float:
        mask = teeth & (sign * MU >= 3) & (np.abs(MU) <= 40)
        values = spec[mask]
        if len(values) < 6:
            return float("nan")
        return float(np.std(np.diff(values, n=2)))

    return {
        "n30": int(np.sum(teeth)),
        "both": both,
        "wing_adj": float(np.mean(wing)) if len(wing) else float("nan"),
        "core_adj": float(np.mean(core)) if len(core) else float("nan"),
        "pump_excess": pump_excess,
        "rough": np.nanmean([rough(-1), rough(1)]),
        "rough_max": np.nanmax([rough(-1), rough(1)]),
    }


def main() -> None:
    table = pd.read_csv(ROOT / "results" / "dark_library_labeled.csv")
    spec = np.load(ROOT / "results" / "dark_library.npz")["spec"]
    rows = []
    for index, row in table.iterrows():
        item = metrics(spec[index])
        item["case"] = int(row["case"])
        item["label"] = row["human_label"]
        rows.append(item)
    frame = pd.DataFrame(rows)
    for label, part in frame.groupby("label"):
        print("\n", label, len(part))
        for column in ("n30", "both", "wing_adj", "core_adj", "pump_excess", "rough", "rough_max"):
            values = part[column].dropna()
            print(f"  {column:12} {values.min():7.3f} {values.median():7.3f} {values.max():7.3f}")
    print("\nFOCUS")
    print(frame[frame["case"].isin([15, 16, 17, 35, 37])].to_string(index=False))
    frame.to_csv(ROOT / "results" / "dark_wing_metrics.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
