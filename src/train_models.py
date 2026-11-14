"""用仿真表训练课程里的分类器。

输入只有泵浦、失谐、色散符号和到达方式（扫描或冷启动）。
标签来自时域波形，不把峰数再喂回模型，避免自己预测自己。
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

STATE_CN = {
    "continuous_wave": "连续光",
    "modulation_instability": "调制不稳定性",
    "chaotic_or_breathing": "混沌或呼吸",
    "single_soliton": "单孤子",
    "multi_soliton": "多孤子",
}


def main():
    frame = pd.read_csv(OUT / "lle_states.csv")
    frame = frame[frame["protocol"].isin(["scan", "cold_start"])].copy()
    frame["scan"] = (frame["protocol"] == "scan").astype(float)
    frame["anomalous"] = (frame["d2"] > 0).astype(float)
    features = frame[["pump", "alpha", "scan", "anomalous"]]
    encoder = LabelEncoder()
    y = encoder.fit_transform(frame["state"])
    x_train, x_test, y_train, y_test = train_test_split(
        features, y, test_size=0.3, random_state=7, stratify=y
    )
    scaler = StandardScaler()
    x_train_s = scaler.fit_transform(x_train)
    x_test_s = scaler.transform(x_test)

    models = {
        "logistic_regression": LogisticRegression(max_iter=800),
        "knn_k5": KNeighborsClassifier(n_neighbors=5),
        "decision_tree": DecisionTreeClassifier(max_depth=4, random_state=7),
    }
    summary = []
    predictions = {}
    for name, model in models.items():
        model.fit(x_train_s, y_train)
        pred = model.predict(x_test_s)
        predictions[name] = pred
        summary.append({
            "model": name,
            "accuracy": float(accuracy_score(y_test, pred)),
            "macro_f1": float(f1_score(y_test, pred, average="macro")),
            "train_rows": int(len(y_train)),
            "test_rows": int(len(y_test)),
        })
        print(name, summary[-1])

    labels = list(encoder.classes_)
    matrix = confusion_matrix(y_test, predictions["decision_tree"], labels=range(len(labels)))
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(labels)), [STATE_CN.get(name, name) for name in labels], rotation=30, ha="right")
    ax.set_yticks(range(len(labels)), [STATE_CN.get(name, name) for name in labels])
    ax.set_xlabel("预测")
    ax.set_ylabel("时域判定")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center", color="black")
    ax.set_title("决策树在测试集上的混淆矩阵")
    fig.colorbar(image, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(OUT / "confusion_tree.png", dpi=160)
    plt.close(fig)

    soliton = frame[(frame["state"] == "single_soliton") & (frame["protocol"] == "scan")]
    if len(soliton) >= 3:
        coeff = np.polyfit(soliton["alpha"], soliton["lines_above_20db"], 1)
        fig, ax = plt.subplots(figsize=(6.2, 4.0))
        ax.scatter(soliton["alpha"], soliton["lines_above_20db"], c="#54A24B")
        xs = np.linspace(soliton["alpha"].min(), soliton["alpha"].max(), 50)
        ax.plot(xs, np.polyval(coeff, xs), color="#333333")
        ax.set_xlabel("归一化失谐 α")
        ax.set_ylabel("高于峰值 20 dB 的模式数")
        ax.set_title("单孤子段：失谐越大，谱越宽")
        fig.tight_layout()
        fig.savefig(OUT / "soliton_bandwidth_regression.png", dpi=160)
        plt.close(fig)
        bandwidth_fit = {"slope_lines_per_alpha": float(coeff[0]), "intercept": float(coeff[1]), "n": int(len(soliton))}
    else:
        bandwidth_fit = None

    payload = {"models": summary, "classes": labels, "bandwidth_fit": bandwidth_fit}
    (OUT / "metrics.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
