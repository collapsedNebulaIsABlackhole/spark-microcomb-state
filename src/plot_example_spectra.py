"""核对第 2 节要用的数字，并画出三类形态的原始光谱。"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
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
ZENODO = (
    ROOT
    / "data"
    / "zenodo-20701147"
    / "Integrated_soliton_microcombs_beyond_the_turnkey_limit_data"
    / "Fig2_soliton_generation"
)
RESULTS = ROOT / "results"
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

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


def stats() -> pd.DataFrame:
    rank = pd.read_csv(RESULTS / "course_method_rank.csv")
    states = pd.read_csv(RESULTS / "public_spectra_states.csv")
    frame = rank.merge(
        states[["split", "file", "mu_span_20db", "peak_power_dbm", "sech_rms", "n_lines"]],
        on=["split", "file"],
        how="left",
    )
    print("COUNTS")
    print(frame["shape"].value_counts().to_string())
    print("BY_SPLIT")
    print(frame.groupby(["split", "shape"]).size().to_string())
    for shape, part in frame.groupby("shape"):
        print(
            f"RANGE {shape} n={len(part)} "
            f"adj {part.adj_db.min():.3f}-{part.adj_db.max():.3f} "
            f"sech {part.sech_rms.min():.3f}-{part.sech_rms.max():.3f} "
            f"span {part.mu_span_20db.min():.1f}-{part.mu_span_20db.max():.1f} "
            f"peak {part.peak_power_dbm.min():.2f}-{part.peak_power_dbm.max():.2f} "
            f"sym_mae {part.sym_mae.min():.2f}-{part.sym_mae.max():.2f} "
            f"sym_corr {part.sym_corr.min():.3f}-{part.sym_corr.max():.3f}"
        )
    wide = frame[frame["shape"].isin(["smooth", "modulated"])].dropna(subset=FEATURES)
    train = wide[wide["split"] == "TrainingSet"]
    scaler = StandardScaler().fit(train[FEATURES])
    y_train = train["shape"].to_numpy()
    logistic = LogisticRegression(max_iter=400).fit(scaler.transform(train[FEATURES]), y_train)
    tree = DecisionTreeClassifier(max_depth=4, random_state=7).fit(scaler.transform(train[FEATURES]), y_train)
    knn = KNeighborsClassifier(n_neighbors=5).fit(scaler.transform(train[FEATURES]), y_train)
    classes = list(logistic.classes_)
    print("CLASSES", classes)
    print("COEF toward", classes[1] if classes[0] == "modulated" else classes[0])
    coef = logistic.coef_[0]
    if classes[0] != "smooth":
        coef = -coef
    for name, value in sorted(zip(FEATURES, coef), key=lambda item: abs(item[1]), reverse=True):
        print(f"  {name} {value:.3f}")
    order = np.argsort(tree.feature_importances_)[::-1]
    print("TREE")
    for index in order:
        print(f"  {FEATURES[index]} {tree.feature_importances_[index]:.3f}")
    for name, model in (("LOGISTIC", logistic), ("TREE", tree), ("KNN5", knn)):
        for split in ("TestingSet", "ValidationSet"):
            part = wide[wide["split"] == split]
            pred = model.predict(scaler.transform(part[FEATURES]))
            print(
                name,
                split,
                len(part),
                f"{accuracy_score(part['shape'], pred):.4f}",
                f"{f1_score(part['shape'], pred, average='macro'):.4f}",
            )
    whole = frame.dropna(subset=FEATURES)
    labels = KMeans(n_clusters=3, n_init=10, random_state=7).fit_predict(
        StandardScaler().fit_transform(whole[FEATURES])
    )
    whole = whole.copy()
    whole["cluster"] = labels
    print("KMEANS")
    print(pd.crosstab(whole["shape"], whole["cluster"]).to_string())
    frequent(frame.dropna(subset=FEATURES), train[FEATURES].median())
    return frame


def frequent(frame: pd.DataFrame, medians: pd.Series) -> None:
    items = []
    for _, row in frame.iterrows():
        bag = {row["shape"]}
        for feature in FEATURES:
            side = "low" if row[feature] <= medians[feature] else "high"
            bag.add(f"{feature}_{side}")
        items.append(bag)
    transactions = pd.Series(items)
    print("RULES")
    candidates = []
    for feature in FEATURES:
        for side in ("low", "high"):
            item = f"{feature}_{side}"
            support = float(transactions.map(lambda bag, item=item: item in bag).mean())
            if support < 0.2:
                continue
            for shape in ("smooth", "modulated", "narrow"):
                both = float(
                    transactions.map(lambda bag, item=item, shape=shape: item in bag and shape in bag).mean()
                )
                confidence = both / support if support else 0.0
                if confidence >= 0.8:
                    candidates.append((confidence, support, item, shape))
    candidates.sort(reverse=True)
    for confidence, support, item, shape in candidates[:12]:
        print(f"  {item} => {shape}  conf {confidence:.3f}  sup {support:.3f}")


def current_windows() -> None:
    for name in ("no", "partial", "complete"):
        path = ZENODO / f"fig2b_step_count_{name}_hybridization.csv"
        frame = pd.read_csv(path)
        power = frame["normalized_comb_power_bin_center"]
        hits = frame["hit_count"]
        frame = frame.assign(
            low=np.where(power < 0.15, hits, 0),
            mid=np.where((power >= 0.15) & (power < 0.65), hits, 0),
            high=np.where(power >= 0.65, hits, 0),
        )
        grouped = frame.groupby("scan_current_mA")[["low", "mid", "high"]].sum()
        total = grouped.sum(axis=1).replace(0, np.nan)
        share = grouped.div(total, axis=0).dropna()
        print(f"CURRENT {name} points {len(share)} I {share.index.min():.2f}-{share.index.max():.2f}")
        for column, threshold in (("high", 0.5), ("mid", 0.5)):
            mask = share[column] >= threshold
            if not mask.any():
                print(f"  {column}>={threshold}: none")
                continue
            idx = share.index.to_numpy()
            selected = idx[mask.to_numpy()]
            breaks = np.where(np.diff(selected) > 0.05)[0]
            start = 0
            for stop in list(breaks) + [len(selected) - 1]:
                segment = selected[start : stop + 1]
                part = share.loc[segment]
                print(
                    f"  {column}>={threshold}: {segment[0]:.2f}-{segment[-1]:.2f} mA "
                    f"mean {part[column].mean():.3f} n={len(segment)}"
                )
                start = stop + 1


def example_spectra(frame: pd.DataFrame) -> None:
    shapes = ("smooth", "modulated", "narrow")
    chosen = {}
    for shape in shapes:
        part = frame[frame["shape"] == shape].copy()
        part["score"] = (part["adj_db"] - part["adj_db"].median()).abs()
        chosen[shape] = list(part.nsmallest(2, "score").itertuples(index=False))
    fig, axes = plt.subplots(2, 3, figsize=(9.2, 5.6), sharey=True)
    titles = {"smooth": "平滑包络", "modulated": "起伏包络", "narrow": "窄谱"}
    colors = {"smooth": "#54A24B", "modulated": "#E45756", "narrow": "#4C78A8"}
    for col, shape in enumerate(shapes):
        for row_index, row in enumerate(chosen[shape]):
            ax = axes[row_index, col]
            path = DATA / row.split / row.file
            spectrum = pd.read_excel(path).dropna(subset=["mu", "power"]).sort_values("mu")
            ax.plot(spectrum["mu"], spectrum["power"], color=colors[shape], lw=1.0)
            ax.set_title(titles[shape], fontsize=10)
            ax.set_xlabel("模式号 $\\mu$")
            ax.tick_params(labelsize=8)
            print("PICK", row.shape, row.split, row.file, f"adj {row.adj_db:.3f}")
    axes[0, 0].set_ylabel("功率 / dBm")
    axes[1, 0].set_ylabel("功率 / dBm")
    fig.tight_layout()
    fig.savefig(RESULTS / "public_example_spectra.png", dpi=160)
    plt.close(fig)
    print("WROTE public_example_spectra.png")


if __name__ == "__main__":
    table = stats()
    example_spectra(table)
    current_windows()
