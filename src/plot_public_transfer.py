"""公开数据的齿间统计图，以及这些特征搬到暗孤子四条谱上的排序。"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

FEATURES = [
    "adj_db",
    "sym_mae",
    "sym_corr",
    "gap_frac",
    "even_share",
    "left_share",
    "pump_excess_db",
    "envelope_corr",
    "sech_rms",
]
LABELS = {
    "adj_db": "相邻齿功率差",
    "sym_mae": "左右对称模式功率差",
    "sym_corr": "左右对称相关性",
    "gap_frac": "齿距不规则比例",
    "even_share": "偶模能量占比",
    "left_share": "左侧能量占比",
    "pump_excess_db": "泵浦高出其余齿",
    "envelope_corr": "功率随远离泵浦的相关",
    "sech_rms": "sech² 包络残差",
}
SHAPE_CN = {"narrow": "窄谱", "smooth": "平滑包络", "modulated": "起伏包络"}
COLORS = {"narrow": "#4C78A8", "smooth": "#54A24B", "modulated": "#E45756"}


def sech_residual(mu: np.ndarray, power_dbm: np.ndarray) -> float:
    keep = np.abs(mu) > 1
    mu_k = mu[keep]
    linear = 10 ** (power_dbm[keep] / 10)
    if len(mu_k) < 8 or np.sum(linear) <= 0:
        return float("nan")
    center = float(np.sum(linear * mu_k) / np.sum(linear))
    offset = mu_k - center
    best = 1e9
    for width in np.linspace(4, 70, 34):
        envelope = 1 / np.cosh(offset / width) ** 2
        denom = float(np.dot(envelope, envelope))
        if denom <= 0:
            continue
        scale = float(np.dot(envelope, linear) / denom)
        error = np.sqrt(np.mean((linear - scale * envelope) ** 2)) / (float(np.mean(linear)) + 1e-30)
        best = min(best, error)
    return float(best)


def symmetry(mu: np.ndarray, power: np.ndarray) -> tuple[float, float]:
    table = {int(round(mode)): value for mode, value in zip(mu, power)}
    left, right = [], []
    for mode, value in table.items():
        if mode > 0 and -mode in table:
            right.append(value)
            left.append(table[-mode])
    if len(left) < 5:
        return float("nan"), float("nan")
    return float(np.mean(np.abs(np.array(left) - np.array(right)))), float(np.corrcoef(left, right)[0, 1])


def user_rows() -> pd.DataFrame:
    rows = []
    names = {
        "spectra_platicon.csv": "单暗脉冲",
        "spectra_cw_locked.csv": "热连续光",
        "spectra_unlocked.csv": "未锁定",
        "spectra_mi_legacy.csv": "调制不稳定",
    }
    for file_name, label in names.items():
        raw = pd.read_csv(ROOT / "data" / "homework_spark" / file_name)
        mu = raw["mu"].to_numpy(float)
        power = np.array([complex(str(text).replace("i", "j")).real for text in raw["dB_norm"]], dtype=float)
        order = np.argsort(mu)
        mu, power = mu[order], power[order]
        adj = np.abs(np.diff(power))
        dmu = np.diff(mu)
        adjacent = adj[np.isclose(dmu, 1)] if np.any(np.isclose(dmu, 1)) else adj
        sym_mae, sym_corr = symmetry(mu, power)
        linear = 10 ** (power / 10)
        left = float(linear[mu < 0].sum())
        right = float(linear[mu > 0].sum())
        even = float(linear[np.round(mu) % 2 == 0].sum())
        odd = float(linear[np.round(mu) % 2 != 0].sum())
        distance = np.abs(mu)
        envelope = float(np.corrcoef(distance, power)[0, 1]) if np.std(power) > 0 else float("nan")
        pump = int(np.argmin(np.abs(mu)))
        rows.append(
            {
                "label": label,
                "adj_db": float(np.mean(adjacent)),
                "sym_mae": sym_mae,
                "sym_corr": sym_corr,
                "gap_frac": float("nan"),
                "even_share": even / (even + odd),
                "left_share": left / (left + right),
                "pump_excess_db": float(power[pump] - np.median(np.delete(power, pump))),
                "envelope_corr": envelope,
                "sech_rms": sech_residual(mu, power),
            }
        )
    return pd.DataFrame(rows)


def box_by_shape(frame: pd.DataFrame, feature: str, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    data = [frame.loc[frame["shape"] == shape, feature].dropna().to_numpy() for shape in ("narrow", "smooth", "modulated")]
    box = ax.boxplot(data, tick_labels=[SHAPE_CN[s] for s in ("narrow", "smooth", "modulated")], showfliers=False, patch_artist=True)
    for patch, shape in zip(box["boxes"], ("narrow", "smooth", "modulated")):
        patch.set_facecolor(COLORS[shape])
        patch.set_alpha(0.85)
    ax.set_ylabel(LABELS[feature])
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    tooth = pd.read_csv(RESULTS / "course_method_rank.csv")
    states = pd.read_csv(RESULTS / "public_spectra_states.csv")[["split", "file", "sech_rms"]]
    frame = tooth.merge(states, on=["split", "file"], how="left")
    own = user_rows()

    box_by_shape(frame, "adj_db", RESULTS / "public_tooth_adj.png", "公开光谱：相邻梳齿功率差")
    box_by_shape(frame, "sym_mae", RESULTS / "public_symmetry.png", "公开光谱：左右对称模式的功率差")
    box_by_shape(frame, "sech_rms", RESULTS / "public_sech_residual.png", "公开光谱：sech² 包络残差")

    smooth = frame[frame["shape"] == "smooth"]
    modulated = frame[frame["shape"] == "modulated"]
    ranks = []
    for feature in FEATURES:
        a = smooth[feature].dropna()
        b = modulated[feature].dropna()
        pooled = np.sqrt(((a.std() ** 2) + (b.std() ** 2)) / 2)
        effect = float((a.mean() - b.mean()) / pooled) if pooled else 0.0
        ranks.append({"feature": feature, "effect_smooth_minus_modulated": effect, "abs_effect": abs(effect)})
    rank = pd.DataFrame(ranks).sort_values("abs_effect", ascending=False)
    rank.to_csv(RESULTS / "public_feature_effect.csv", index=False)
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.barh([LABELS[item] for item in rank["feature"][::-1]], rank["abs_effect"][::-1], color="#4C78A8")
    ax.set_xlabel("平滑包络和起伏包络的分离程度")
    ax.set_title("公开亮孤子：机器排出的特征顺序")
    fig.tight_layout()
    fig.savefig(RESULTS / "public_feature_rank.png", dpi=160)
    plt.close(fig)

    print("RANK")
    for index, row in enumerate(rank.itertuples(index=False), start=1):
        print(index, row.feature, round(row.effect_smooth_minus_modulated, 3))
    print("OWN")
    print(own.round(3).to_string(index=False))
    own.to_csv(RESULTS / "user_transfer_features.csv", index=False)
    print("PLOTS_DONE")


if __name__ == "__main__":
    main()
