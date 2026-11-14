"""工作点表上的课程作业分析：分组、台阶检验、无泄漏分类和回归。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "homework_spark" / "operating_points.csv"
OUT = ROOT / "results"


def main() -> None:
    frame = pd.read_csv(DATA)
    frame["rev_flag"] = (frame["detuning_mode"] == "rev_memory").astype(int)

    grouped = (
        frame.groupby(["state", "detuning_mode"], as_index=False)
        .agg(
            n=("Pep", "size"),
            pep_avg=("Pep", "mean"),
            pep_min=("Pep", "min"),
            pep_max=("Pep", "max"),
            dark_avg=("dark_frac", "mean"),
            alpha_min=("alpha_L", "min"),
            alpha_max=("alpha_L", "max"),
            n_dark_max=("n_dark", "max"),
        )
        .sort_values(["state", "detuning_mode"])
    )
    grouped.to_csv(OUT / "state_summary.csv", index=False)

    platicon = frame[(frame["state"] == "platicon") & (frame["detuning_mode"] == "rev_memory")].sort_values("alpha_L")
    dark_slope, dark_intercept = np.polyfit(platicon["alpha_L"], platicon["dark_frac"], 1)
    pep_slope, pep_intercept = np.polyfit(platicon["alpha_L"], platicon["Pep"], 1)
    dark_resid = platicon["dark_frac"] - (dark_slope * platicon["alpha_L"] + dark_intercept)
    pep_resid = platicon["Pep"] - (pep_slope * platicon["alpha_L"] + pep_intercept)
    dark_jump = platicon["dark_frac"].diff().abs()
    pep_jump = platicon["Pep"].diff().abs()

    features = frame[["alpha_L", "rev_flag"]]
    labels = frame["state"]
    x_train, x_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.3, random_state=7, stratify=labels
    )
    scaler = StandardScaler()
    x_train_s = scaler.fit_transform(x_train)
    x_test_s = scaler.transform(x_test)
    logistic = LogisticRegression(max_iter=500)
    logistic.fit(x_train_s, y_train)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(x_train_s, y_train)
    pred_lr = logistic.predict(x_test_s)
    pred_knn = knn.predict(x_test_s)

    reg_x = platicon[["alpha_L"]]
    reg = LinearRegression().fit(reg_x, platicon["Pep"])

    report = {
        "rows": int(len(frame)),
        "train_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
        "features": ["alpha_L", "rev_flag"],
        "excluded_because_they_define_the_label": ["Pep", "Pem", "n_peak", "n_dip", "n_dark", "dark_frac", "contrast"],
        "logistic_accuracy": float(accuracy_score(y_test, pred_lr)),
        "logistic_macro_f1": float(f1_score(y_test, pred_lr, average="macro")),
        "knn_accuracy": float(accuracy_score(y_test, pred_knn)),
        "knn_macro_f1": float(f1_score(y_test, pred_knn, average="macro")),
        "platicon_points": int(len(platicon)),
        "platicon_n_dark_unique": sorted(platicon["n_dark"].unique().tolist()),
        "platicon_n_dip_counts": {str(k): int(v) for k, v in platicon["n_dip"].value_counts().sort_index().items()},
        "dark_frac_slope_per_alpha": float(dark_slope),
        "dark_frac_residual_std": float(dark_resid.std()),
        "dark_frac_residual_max_abs": float(dark_resid.abs().max()),
        "dark_frac_jumps_over_0_02": int((dark_jump > 0.02).sum()),
        "pep_min": float(platicon["Pep"].min()),
        "pep_max": float(platicon["Pep"].max()),
        "pep_std": float(platicon["Pep"].std()),
        "pep_slope_per_alpha": float(pep_slope),
        "pep_r2": float(r2_score(platicon["Pep"], reg.predict(reg_x))),
        "pep_residual_std": float(pep_resid.std()),
        "pep_max_consecutive_jump": float(pep_jump.max()),
        "alpha_span": [float(platicon["alpha_L"].min()), float(platicon["alpha_L"].max())],
    }
    (OUT / "operating_point_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(grouped.to_string(index=False))


if __name__ == "__main__":
    main()
