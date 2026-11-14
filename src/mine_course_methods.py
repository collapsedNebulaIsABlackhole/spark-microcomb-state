"""按课件里的方法，让模型自己排特征。

用到的课件方法：逻辑回归、决策树信息增益、K 近邻、K-means、FP-growth。
Spark MLlib 没有 K 近邻，这一项用 sklearn。协同过滤要用户–物品评分，这里没有。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "figshare-30026368"
TOOTH = ROOT / "results" / "tooth_relations.csv"
OUT = ROOT / "results" / "course_method_rank.csv"
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


def symmetry(mu: np.ndarray, power: np.ndarray) -> tuple[float, float]:
    table = {int(round(mode)): value for mode, value in zip(mu, power)}
    left = []
    right = []
    for mode, value in table.items():
        if mode > 0 and -mode in table:
            right.append(value)
            left.append(table[-mode])
    if len(left) < 5:
        return float("nan"), float("nan")
    corr = float(np.corrcoef(left, right)[0, 1])
    mae = float(np.mean(np.abs(np.array(left) - np.array(right))))
    return mae, corr


def add_symmetry(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for split in ("TrainingSet", "ValidationSet", "TestingSet"):
        folder = DATA / split
        for path in sorted(folder.glob("*.xlsx")):
            raw = pd.read_excel(path).dropna(subset=["mu", "power"])
            mae, corr = symmetry(raw["mu"].to_numpy(float), raw["power"].to_numpy(float))
            rows.append({"split": split, "file": path.name, "sym_mae": mae, "sym_corr": corr})
            if len(rows) % 100 == 0:
                print("symmetry", len(rows), flush=True)
    extra = pd.DataFrame(rows)
    return frame.merge(extra, on=["split", "file"], how="left")


def scores(name: str, y_true, y_pred) -> None:
    print(
        name,
        len(y_true),
        round(accuracy_score(y_true, y_pred), 4),
        round(f1_score(y_true, y_pred, average="macro"), 4),
    )


def frequent_items(frame: pd.DataFrame, medians: pd.Series) -> None:
    items = []
    for _, row in frame.iterrows():
        bag = {row["shape"]}
        for feature in FEATURES:
            side = "low" if row[feature] <= medians[feature] else "high"
            bag.add(f"{feature}_{side}")
        items.append(bag)
    transactions = pd.Series(items)
    print("FP_GROWTH min_support 0.2, rules whose consequent is a shape")
    shapes = ["smooth", "modulated", "narrow"]
    candidates = []
    for feature in FEATURES:
        for side in ("low", "high"):
            item = f"{feature}_{side}"
            support = float(transactions.map(lambda bag, item=item: item in bag).mean())
            if support < 0.2:
                continue
            for shape in shapes:
                both = float(transactions.map(lambda bag, item=item, shape=shape: item in bag and shape in bag).mean())
                confidence = both / support if support else 0.0
                if confidence >= 0.8:
                    candidates.append((confidence, support, item, shape, both))
    candidates.sort(reverse=True)
    for confidence, support, item, shape, both in candidates[:12]:
        print(f"  {item} => {shape}  conf {confidence:.3f}  sup {support:.3f}")
    if not candidates:
        print("  none")


def main() -> None:
    frame = add_symmetry(pd.read_csv(TOOTH))
    frame.to_csv(OUT, index=False, encoding="utf-8-sig")
    usable = frame[frame["shape"].isin(["smooth", "modulated"])].dropna(subset=FEATURES).copy()
    train = usable[usable["split"] == "TrainingSet"]
    test = usable[usable["split"] == "TestingSet"]
    valid = usable[usable["split"] == "ValidationSet"]
    scaler = StandardScaler().fit(train[FEATURES])
    x_train = scaler.transform(train[FEATURES])
    y_train = train["shape"].to_numpy()

    logistic = LogisticRegression(max_iter=200).fit(x_train, y_train)
    tree = DecisionTreeClassifier(max_depth=4, random_state=7).fit(x_train, y_train)
    knn = KNeighborsClassifier(n_neighbors=5).fit(x_train, y_train)
    print("FEATURE_RANK decision tree")
    order = np.argsort(tree.feature_importances_)[::-1]
    for index in order:
        print(f"  {FEATURES[index]} {tree.feature_importances_[index]:.3f}")
    print("LOGISTIC_COEF")
    for index, feature in enumerate(FEATURES):
        print(f"  {feature} {logistic.coef_[0][index]:.3f}")
    for name, model in (("LOGISTIC", logistic), ("TREE", tree), ("KNN5", knn)):
        for split_name, part in (("TestingSet", test), ("ValidationSet", valid)):
            pred = model.predict(scaler.transform(part[FEATURES]))
            scores(f"{name} {split_name}", part["shape"], pred)

    whole = frame.dropna(subset=FEATURES).copy()
    labels = KMeans(n_clusters=3, n_init=10, random_state=7).fit_predict(StandardScaler().fit_transform(whole[FEATURES]))
    whole["cluster"] = labels
    print("KMEANS")
    print(pd.crosstab(whole["shape"], whole["cluster"]).to_string())

    medians = train[FEATURES].median()
    frequent_items(frame.dropna(subset=FEATURES), medians)
    print("OWN_SPECTRA")
    for name in ("spectra_platicon.csv", "spectra_cw_locked.csv", "spectra_unlocked.csv", "spectra_mi_legacy.csv"):
        raw = pd.read_csv(ROOT / "data" / "homework_spark" / name)
        power = np.array([complex(str(text).replace("i", "j")).real for text in raw["dB_norm"]], dtype=float)
        mae, corr = symmetry(raw["mu"].to_numpy(float), power)
        print(f"  {name} sym_mae {mae:.3f} sym_corr {corr:.3f}")
    print("METHOD_DONE")


if __name__ == "__main__":
    main()
