"""逐条检验暗孤子光谱筛选规则。时域类别只作最终核对，规则本身只读光谱。"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
MU = np.arange(-128, 128, dtype=float)


def pair_diffs(spec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    peak = float(np.max(spec))
    teeth = np.where(spec > peak - 35)[0]
    mu_t = MU[teeth]
    power = spec[teeth]
    order = np.argsort(mu_t)
    mu_t, power = mu_t[order], power[order]
    step = np.diff(mu_t)
    adjacent = np.isclose(step, 1)
    left = mu_t[:-1][adjacent]
    delta = np.abs(np.diff(power)[adjacent])
    return left, delta


def symmetry(spec: np.ndarray) -> tuple[float, float]:
    peak = float(np.max(spec))
    table = {int(mode): float(value) for mode, value in zip(MU, spec) if value > peak - 35}
    left, right = [], []
    for mode, value in table.items():
        if mode > 0 and -mode in table:
            right.append(value)
            left.append(table[-mode])
    if len(left) < 5:
        return float("nan"), float("nan")
    return float(np.mean(np.abs(np.array(left) - np.array(right)))), float(np.corrcoef(left, right)[0, 1])


def wing_slopes(spec: np.ndarray) -> tuple[float, float]:
    peak = float(np.max(spec))
    loud = spec > peak - 35

    def slope(sign: int) -> float:
        mask = loud & (sign * MU >= 2) & (np.abs(MU) <= 45)
        if np.count_nonzero(mask) < 4 or float(np.std(MU[mask])) < 1e-6:
            return float("nan")
        return float(np.polyfit(np.abs(MU[mask]), spec[mask], 1)[0])

    return slope(-1), slope(1)


def features_of(spec: np.ndarray, ref: np.ndarray) -> dict:
    left, delta = pair_diffs(spec)
    wing = np.abs(left) >= 2
    core = np.abs(left) <= 1
    sym_mae, sym_corr = symmetry(spec)
    slope_l, slope_r = wing_slopes(spec)
    band = (np.abs(MU) >= 2) & (np.abs(MU) <= 45) & (ref > np.max(ref) - 40)
    corr = float("nan")
    if np.count_nonzero(band) >= 8 and float(np.std(spec[band])) > 1e-6 and float(np.std(ref[band])) > 1e-6:
        corr = float(np.corrcoef(spec[band], ref[band])[0, 1])
    return {
        "global_adj": float(np.mean(delta)) if len(delta) else float("nan"),
        "wing_adj": float(np.mean(delta[wing])) if np.any(wing) else float("nan"),
        "core_adj": float(np.mean(delta[core])) if np.any(core) else float("nan"),
        "sym_mae": sym_mae,
        "sym_corr": sym_corr,
        "slope_l": slope_l,
        "slope_r": slope_r,
        "slope_gap": abs(slope_l - slope_r) if np.isfinite(slope_l) and np.isfinite(slope_r) else float("nan"),
        "template_corr": corr,
        "n_pairs": int(len(delta)),
    }


def binary_scores(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "n_pass": int(np.sum(y_pred)),
        "n_hit": int(np.sum(y_pred & y_true)),
    }


def best_threshold(values: np.ndarray, y_true: np.ndarray, higher_is_dark: bool) -> tuple[float, dict]:
    usable = np.isfinite(values)
    candidates = np.unique(np.round(values[usable], 4))
    best = (-1.0, float("nan"), {})
    for cut in candidates:
        pred = values >= cut if higher_is_dark else values <= cut
        pred = pred & usable
        score = binary_scores(y_true, pred)
        if score["f1"] > best[0]:
            best = (score["f1"], float(cut), score)
    return best[1], best[2]


def main() -> None:
    table = pd.read_csv(RESULTS / "dark_library.csv")
    pack = np.load(RESULTS / "dark_library.npz")
    spec = pack["spec"]
    ref = pack["ref"]
    rows = [features_of(spec[i], ref) for i in range(len(table))]
    feat = pd.concat([table, pd.DataFrame(rows)], axis=1)
    feat["dark"] = feat["time_kind"].eq("暗孤子")
    feat.to_csv(RESULTS / "dark_screen_features.csv", index=False, encoding="utf-8-sig")
    y = feat["dark"].to_numpy()
    print("N", len(feat), "dark", int(y.sum()), "other", int((~y).sum()))
    for column in ("global_adj", "wing_adj", "core_adj", "sym_corr", "sym_mae", "template_corr", "slope_gap"):
        for flag, name in ((True, "dark"), (False, "other")):
            part = feat.loc[feat["dark"] == flag, column].dropna()
            print(
                f"{column:14} {name:5} n={len(part):2d} "
                f"min={part.min():7.3f} med={part.median():7.3f} max={part.max():7.3f}"
            )

    ref_row = feat[(feat["AMX_MHz"] == 0) & np.isclose(feat["P_pump"], 0.2) & np.isclose(feat["delw"], 0.060125)]
    anchor = ref_row.iloc[0]
    print("ANCHOR", {key: round(float(anchor[key]), 3) for key in ("global_adj", "wing_adj", "core_adj", "sym_corr", "sym_mae", "template_corr", "slope_gap")})

    rules = {
        "全局相邻齿差": feat["global_adj"] <= 1.5 * float(anchor["global_adj"]),
        "翼部分段齿差": (feat["wing_adj"] <= 1.5 * float(anchor["wing_adj"])) & (feat["wing_adj"] >= 0.4 * float(anchor["wing_adj"])),
        "翼状模板": feat["template_corr"] >= 0.85,
        "左右对称": (feat["sym_corr"] >= 0.90) & (feat["sym_mae"] <= 2 * float(anchor["sym_mae"])),
    }
    rules["分段+模板+对称"] = rules["翼部分段齿差"] & rules["翼状模板"] & rules["左右对称"]
    summary = {}
    for name, pred in rules.items():
        pred = pred.fillna(False).to_numpy(bool)
        summary[name] = {"threshold": "anchor", **binary_scores(y, pred)}
        print("RULE", name, summary[name])

    tuned = {}
    for name, column, higher in (
        ("全局相邻齿差-扫描阈值", "global_adj", False),
        ("翼部分段齿差-扫描阈值", "wing_adj", False),
        ("翼状模板-扫描阈值", "template_corr", True),
        ("对称相关-扫描阈值", "sym_corr", True),
    ):
        cut, score = best_threshold(feat[column].to_numpy(float), y, higher)
        tuned[name] = {"cut": cut, "higher": higher, **score}
        print("TUNE", name, tuned[name])

    model_cols = ["wing_adj", "template_corr", "sym_corr", "sym_mae", "slope_gap"]
    usable = feat.dropna(subset=model_cols).copy()
    pred = np.zeros(len(usable), dtype=bool)
    labels = usable["dark"].to_numpy()
    matrix = usable[model_cols].to_numpy(float)
    for index in range(len(usable)):
        train = np.ones(len(usable), dtype=bool)
        train[index] = False
        if labels[train].sum() == 0 or (~labels[train]).sum() == 0:
            continue
        scaler = StandardScaler().fit(matrix[train])
        model = LogisticRegression(max_iter=400).fit(scaler.transform(matrix[train]), labels[train])
        pred[index] = bool(model.predict(scaler.transform(matrix[index : index + 1]))[0])
    loo = binary_scores(labels, pred)
    print("LOO", loo, "usable", len(usable))

    grid = feat[
        np.isclose(feat["AMX_MHz"], 200)
        & np.isclose(feat["p_cross"], 1)
        & np.isclose(feat["loss_dBcm"], 0.04)
        & np.isclose(feat["seta"], 0.011)
        & np.isclose(feat["amx_sign"], 1)
        & (feat["beta2_ps2_per_km"] > 0)
    ].copy()
    passed = grid[rules["分段+模板+对称"].reindex(grid.index).fillna(False).to_numpy(bool) & grid["dark"].to_numpy()]
    print("GRID", len(grid), "stable_and_pass", len(passed))
    if len(passed):
        print(
            "INTERVAL delw",
            round(float(passed["delw"].min()), 3),
            round(float(passed["delw"].max()), 3),
            "P",
            round(float(passed["P_pump"].min()), 2),
            round(float(passed["P_pump"].max()), 2),
        )
        print(passed[["delw", "P_pump", "template_corr", "wing_adj", "sym_corr", "time_view"]].to_string(index=False))

    query_cols = ["wing_adj", "template_corr", "sym_corr", "slope_gap"]
    library = feat.dropna(subset=query_cols)
    dark_lib = library[library["dark"]]
    scaler = StandardScaler().fit(library[query_cols])
    neighbors = NearestNeighbors(n_neighbors=min(5, len(dark_lib))).fit(scaler.transform(dark_lib[query_cols]))
    cover = 0
    checked = 0
    for position in range(len(library)):
        row = library.iloc[position]
        if not bool(row["dark"]):
            continue
        pool = dark_lib.drop(index=row.name)
        if len(pool) < 3:
            continue
        local = StandardScaler().fit(library.drop(index=row.name)[query_cols])
        model = NearestNeighbors(n_neighbors=3).fit(local.transform(pool[query_cols]))
        distance, index = model.kneighbors(local.transform(row[query_cols].to_numpy(float).reshape(1, -1)))
        near = pool.iloc[index[0]]
        checked += 1
        inside = (
            near["delw"].min() - 0.02 <= row["delw"] <= near["delw"].max() + 0.02
            and near["P_pump"].min() * 0.75 <= row["P_pump"] <= near["P_pump"].max() * 1.25
        )
        cover += int(inside)
    print("LOO_INTERVAL", cover, "/", checked)

    summary["logistic_loo"] = loo
    summary["tuned"] = tuned
    summary["loo_interval"] = {"cover": cover, "checked": checked}
    (RESULTS / "dark_screen_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for flag, color, label, marker in (
        (True, "#1f4e79", "暗孤子", "o"),
        (False, "#E45756", "非暗孤子", "X"),
    ):
        part = grid[grid["dark"] == flag]
        ax.scatter(part["delw"], part["P_pump"], c=color, marker=marker, s=42, label=label, zorder=2)
    if len(passed):
        ax.scatter(
            passed["delw"],
            passed["P_pump"],
            facecolors="none",
            edgecolors="#54A24B",
            s=140,
            linewidths=1.4,
            label="规则通过且时域确认",
            zorder=3,
        )
    ax.set_xlabel(r"归一化失谐 $\delta$")
    ax.set_ylabel("泵浦功率 / W")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS / "dark_param_interval.png", dpi=160)
    plt.close(fig)
    print("WROTE figures and summary")


if __name__ == "__main__":
    main()
