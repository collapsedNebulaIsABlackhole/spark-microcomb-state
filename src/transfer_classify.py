"""把第 2 节已经拟合的分类器原样用到九条光谱上。

标准化只用亮孤子训练集的均值和标准差，不重新估计，也不改分界面。
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io as sio
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
MID = Path(r"C:\Work\大四上\大数据处理\middle")
FEATURES = [
    "adj_db",
    "left_share",
    "even_share",
    "gap_frac",
    "pump_excess_db",
    "envelope_corr",
    "sym_mae",
    "sym_corr",
]
FEATURE_NAME = {
    "adj_db": "相邻齿功率差",
    "left_share": "左侧能量占比",
    "even_share": "偶模能量占比",
    "gap_frac": "齿距不规则比例",
    "pump_excess_db": "泵浦超出量",
    "envelope_corr": "远离泵浦的相关",
    "sym_mae": "对称功率差",
    "sym_corr": "对称相关",
}
LABEL_NAME = {"smooth": "平滑包络", "modulated": "起伏包络"}


def gap_fraction(mu: np.ndarray, power: np.ndarray) -> float:
    """与公开光谱相同：相邻齿的间距超过中位间距 1.5 倍的比例。仿真按模式号等间隔。"""
    peak = float(np.max(power))
    teeth = power > peak - 35
    mu_t = np.sort(mu[teeth])
    steps = np.diff(mu_t)
    steps = steps[steps > 1e-6]
    if len(steps) == 0:
        return float("nan")
    base = float(np.median(steps))
    return float(np.mean(steps > 1.5 * base))


def fit_public():
    frame = pd.read_csv(RESULTS / "course_method_rank.csv")
    usable = frame[frame["shape"].isin(["smooth", "modulated"])].dropna(subset=FEATURES).copy()
    train = usable[usable["split"] == "TrainingSet"]
    scaler = StandardScaler().fit(train[FEATURES])
    x_train = scaler.transform(train[FEATURES])
    y_train = train["shape"].to_numpy()
    logistic = LogisticRegression(max_iter=200).fit(x_train, y_train)
    tree = DecisionTreeClassifier(max_depth=4, random_state=7).fit(x_train, y_train)
    knn = KNeighborsClassifier(n_neighbors=5).fit(x_train, y_train)
    positive = list(logistic.classes_).index("smooth")
    print("classes", list(logistic.classes_), "smooth_row", positive)
    coef = logistic.coef_[0] if logistic.classes_[1] == "smooth" else -logistic.coef_[0]
    for feature, value in zip(FEATURES, coef):
        print(f"  coef {feature} {value:.3f}")
    order = np.argsort(tree.feature_importances_)[::-1]
    for index in order:
        print(f"  tree {FEATURES[index]} {tree.feature_importances_[index]:.3f}")
    return scaler, logistic, tree, knn


def dark_features(scaler: StandardScaler) -> pd.DataFrame:
    middle = pd.read_csv(RESULTS / "middle_transfer_features.csv")
    saved = sio.loadmat(MID / "failed_spectra.mat", squeeze_me=True, struct_as_record=False)
    mu = np.arange(-128, 128, dtype=float)
    stored = {str(item.name): np.asarray(item.spec_dB, float) for item in saved["rec"]}
    rows = []
    for _, row in middle.iterrows():
        spec = stored[row["name"]]
        item = row.to_dict()
        item["gap_frac"] = gap_fraction(mu, spec)
        values = np.array([item[feature] for feature in FEATURES], dtype=float)
        item["complete"] = bool(np.isfinite(values).all())
        if item["complete"]:
            scaled = scaler.transform(values.reshape(1, -1))[0]
            for feature, value in zip(FEATURES, scaled):
                item[f"z_{feature}"] = float(value)
        rows.append(item)
    return pd.DataFrame(rows)


def predict(frame: pd.DataFrame, logistic, tree, knn) -> pd.DataFrame:
    out = frame.copy()
    for name in ("logistic", "tree", "knn"):
        out[name] = ""
    complete = out["complete"].to_numpy()
    if not np.any(complete):
        return out
    scaled = out.loc[complete, [f"z_{feature}" for feature in FEATURES]].to_numpy(float)
    out.loc[complete, "logistic"] = logistic.predict(scaled)
    out.loc[complete, "tree"] = tree.predict(scaled)
    out.loc[complete, "knn"] = knn.predict(scaled)
    return out


def draw(frame: pd.DataFrame) -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    order = frame.copy()
    matrix = []
    for _, row in order.iterrows():
        if row["complete"]:
            matrix.append([row[f"z_{feature}"] for feature in FEATURES])
        else:
            matrix.append([np.nan] * len(FEATURES))
    matrix = np.asarray(matrix, dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.6), gridspec_kw={"width_ratios": [1.55, 1]})
    ax = axes[0]
    limit = np.nanmax(np.abs(matrix))
    image = ax.imshow(matrix, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
    ax.set_xticks(range(len(FEATURES)))
    ax.set_xticklabels([FEATURE_NAME[feature] for feature in FEATURES], rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order["title"].tolist(), fontsize=8)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            if np.isfinite(value):
                ax.text(j, i, f"{value:.1f}", ha="center", va="center", fontsize=7, color="#222222")
    ax.set_title("按训练集均值和标准差标准化", fontsize=10)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[1]
    columns = ("logistic", "tree", "knn")
    headers = ("逻辑回归", "决策树", "K 近邻")
    color = {"smooth": "#54A24B", "modulated": "#E45756", "": "#f0f0f0"}
    for index, (_, row) in enumerate(order.iterrows()):
        for col, key in enumerate(columns):
            label = row[key]
            ax.add_patch(plt.Rectangle((col, len(order) - index - 1), 1, 1, facecolor=color[label], edgecolor="white", lw=1.2))
            ax.text(col + 0.5, len(order) - index - 0.5, LABEL_NAME.get(label, "未计算"), ha="center", va="center", fontsize=7.5, color="#222222")
    ax.set_xlim(0, 3)
    ax.set_ylim(0, len(order))
    ax.set_xticks([0.5, 1.5, 2.5])
    ax.set_xticklabels(headers, fontsize=8)
    ax.set_yticks([len(order) - index - 0.5 for index in range(len(order))])
    ax.set_yticklabels(order["title"].tolist(), fontsize=8)
    ax.tick_params(length=0)
    ax.set_title("原模型给出的类别", fontsize=10)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    fig.savefig(RESULTS / "middle_transfer_classify.png", dpi=160)
    plt.close(fig)


def main() -> None:
    scaler, logistic, tree, knn = fit_public()
    frame = predict(dark_features(scaler), logistic, tree, knn)
    keep = ["name", "title", "time_kind", "time_view", "complete", "logistic", "tree", "knn", *FEATURES]
    keep += [f"z_{feature}" for feature in FEATURES]
    frame[keep].to_csv(RESULTS / "middle_transfer_classify.csv", index=False, encoding="utf-8-sig")
    draw(frame)
    print(frame[["title", "time_kind", "complete", "logistic", "tree", "knn"]].to_string(index=False))
    print("scaler_mean", dict(zip(FEATURES, np.round(scaler.mean_, 4))))
    print("scaler_scale", dict(zip(FEATURES, np.round(scaler.scale_, 4))))
    print("WROTE middle_transfer_classify.csv middle_transfer_classify.png")


if __name__ == "__main__":
    main()
