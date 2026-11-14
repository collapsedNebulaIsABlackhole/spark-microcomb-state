"""公开数据上的主体：Spark 做状态识别和工作点回归。

Figshare 870 条光谱按补充材料的 sech^2 吻合度分成平滑包络和起伏包络。
特征里不含拟合残差。训练集、验证集、测试集沿用数据作者的划分。

Zenodo 10.5281/zenodo.20701147 提供加热功率到谱宽的回归，以及三种模式耦合下的梳功率统计。
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import StandardScaler, StringIndexer, VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit, sum as spark_sum, when

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
PUBLIC = ROOT / "data" / "zenodo-20701147" / "Integrated_soliton_microcombs_beyond_the_turnkey_limit_data"
FEATURES = RESULTS / "public_spectra_features.csv"
STAGE = Path(r"C:\spark-tmp\public")
SPAN_MIN = 50.0
RMS_MAX = 0.50

def spark_path(path: Path) -> str:
    return "file:///" + path.as_posix()


plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def assign_shape(frame: pd.DataFrame) -> pd.DataFrame:
    wide = frame["mu_span_20db"] >= SPAN_MIN
    smooth = frame["sech_rms"] < RMS_MAX
    frame = frame.copy()
    frame["shape"] = np.where(~wide, "narrow", np.where(smooth, "smooth", "modulated"))
    return frame


def macro_f1(pairs: list[tuple[float, float]]) -> float:
    labels = sorted({label for label, _ in pairs})
    scores = []
    for label in labels:
        tp = sum(got == label and pred == label for got, pred in pairs)
        fp = sum(got != label and pred == label for got, pred in pairs)
        fn = sum(got == label and pred != label for got, pred in pairs)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall))
    return float(sum(scores) / len(scores)) if scores else 0.0


def classify(spark: SparkSession, labeled: pd.DataFrame) -> dict:
    usable = labeled[labeled["shape"].isin(["smooth", "modulated"])].copy()
    STAGE.mkdir(parents=True, exist_ok=True)
    usable_path = STAGE / "shapes.csv"
    usable.to_csv(usable_path, index=False)
    frame = spark.read.option("header", True).option("inferSchema", True).csv(spark_path(usable_path))
    indexed = StringIndexer(inputCol="shape", outputCol="label").fit(frame)
    indexed_frame = indexed.transform(frame)

    def one_model(columns: list[str]) -> list[dict]:
        assembled = VectorAssembler(inputCols=columns, outputCol="raw_features").transform(indexed_frame)
        train_raw = assembled.filter(col("split") == lit("TrainingSet"))
        scaler = StandardScaler(inputCol="raw_features", outputCol="features", withMean=True, withStd=True).fit(train_raw)
        model = LogisticRegression(featuresCol="features", labelCol="label", maxIter=100).fit(scaler.transform(train_raw))
        scores = []
        for name in ("TestingSet", "ValidationSet"):
            rows = model.transform(scaler.transform(assembled.filter(col("split") == lit(name)))).select("label", "prediction").collect()
            pairs = [(row.label, row.prediction) for row in rows]
            correct = sum(got == pred for got, pred in pairs)
            item = {
                "split": name,
                "n": len(pairs),
                "accuracy": correct / len(pairs) if pairs else 0.0,
                "macro_f1": macro_f1(pairs),
            }
            print("CLASS", columns, item)
            scores.append(item)
        return scores

    smooth_peak = usable.loc[usable["shape"] == "smooth", "peak_power_dbm"]
    modulated_peak = usable.loc[usable["shape"] == "modulated", "peak_power_dbm"]
    print("LABELS", indexed.labels)
    return {
        "indexer_labels": indexed.labels,
        "label_source": "sech_rms threshold, not an author-published soliton number",
        "with_peak_power": one_model(["lines_within_20db", "mu_span_20db", "peak_power_dbm", "mu_centroid"]),
        "without_peak_power": one_model(["lines_within_20db", "mu_span_20db", "mu_centroid"]),
        "peak_power_dbm_ranges": {
            "smooth": [float(smooth_peak.min()), float(smooth_peak.max())],
            "modulated": [float(modulated_peak.min()), float(modulated_peak.max())],
            "overlap": bool(smooth_peak.max() >= modulated_peak.min() and modulated_peak.max() >= smooth_peak.min()),
        },
    }


def heater_fit(spark: SparkSession) -> dict:
    source = PUBLIC / "Fig3_tuning_and_characterization" / "fig3e_bandwidth_and_linewidth_vs_AR_heater.csv"
    path = STAGE / "heater.csv"
    path.write_bytes(source.read_bytes())
    frame = spark.read.option("header", True).option("inferSchema", True).csv(spark_path(path))
    assembled = VectorAssembler(inputCols=["AR_heater_power_mW"], outputCol="features").transform(frame)
    bandwidth = LinearRegression(featuresCol="features", labelCol="spectral_envelope_bandwidth_3dB_nm", maxIter=50).fit(assembled)
    linewidth = LinearRegression(featuresCol="features", labelCol="lorentzian_linewidth_Hz", maxIter=50).fit(assembled)
    result = {
        "n": frame.count(),
        "bandwidth_slope_nm_per_mW": float(bandwidth.coefficients[0]),
        "bandwidth_intercept_nm": float(bandwidth.intercept),
        "bandwidth_r2": float(bandwidth.summary.r2),
        "linewidth_slope_Hz_per_mW": float(linewidth.coefficients[0]),
        "linewidth_intercept_Hz": float(linewidth.intercept),
        "linewidth_r2": float(linewidth.summary.r2),
    }
    print("HEATER", result)
    table = pd.read_csv(path)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.scatter(table["AR_heater_power_mW"], table["spectral_envelope_bandwidth_3dB_nm"], s=36, c="#4C78A8")
    xs = np.linspace(table["AR_heater_power_mW"].min(), table["AR_heater_power_mW"].max(), 50)
    ax.plot(xs, result["bandwidth_intercept_nm"] + result["bandwidth_slope_nm_per_mW"] * xs, color="#333333")
    ax.set_xlabel("加热功率 / mW")
    ax.set_ylabel("3 dB 谱宽 / nm")
    ax.set_title("公开数据：加热功率预测孤子谱宽")
    fig.tight_layout()
    fig.savefig(RESULTS / "public_heater_bandwidth.png", dpi=160)
    plt.close(fig)
    return result


def hybridization(spark: SparkSession) -> pd.DataFrame:
    folder = PUBLIC / "Fig2_soliton_generation"
    pieces = []
    for name in ("no", "partial", "complete"):
        source = folder / f"fig2b_step_count_{name}_hybridization.csv"
        path = STAGE / source.name
        if not path.exists():
            path.write_bytes(source.read_bytes())
        part = spark.read.option("header", True).option("inferSchema", True).csv(spark_path(path))
        pieces.append(part.withColumn("condition", lit(name)))
    frame = pieces[0]
    for part in pieces[1:]:
        frame = frame.unionByName(part)
    power = col("normalized_comb_power_bin_center")
    hits = col("hit_count")
    grouped = frame.groupBy("condition").agg(
        spark_sum(hits).alias("hits"),
        spark_sum(when(power < 0.15, hits).otherwise(0)).alias("low"),
        spark_sum(when((power >= 0.15) & (power < 0.65), hits).otherwise(0)).alias("mid"),
        spark_sum(when(power >= 0.65, hits).otherwise(0)).alias("high"),
    )
    table = grouped.toPandas()
    for column in ("low", "mid", "high"):
        table[column] = table[column] / table["hits"]
    order = {"no": 0, "partial": 1, "complete": 2}
    table["ord"] = table["condition"].map(order)
    table = table.sort_values("ord").drop(columns="ord")
    table.to_csv(RESULTS / "public_hybridization_levels.csv", index=False)
    print(table.to_string(index=False))

    labels = {"no": "无耦合", "partial": "部分耦合", "complete": "完全耦合"}
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    x = np.arange(len(table))
    low = table["low"].to_numpy()
    mid = table["mid"].to_numpy()
    high = table["high"].to_numpy()
    ax.bar(x, low, color="#4C78A8", label="梳功率接近 0")
    ax.bar(x, mid, bottom=low, color="#F58518", label="中间功率")
    ax.bar(x, high, bottom=low + mid, color="#54A24B", label="高梳功率")
    ax.set_xticks(x, [labels[item] for item in table["condition"]])
    ax.set_ylabel("扫描命中占比")
    ax.set_title("公开扫描：三种耦合各自落在哪一档梳功率")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS / "public_hybridization_levels.png", dpi=160)
    plt.close(fig)
    return table


def phase_plot() -> None:
    path = PUBLIC / "Fig1_principle_simulation" / "fig1c_phase_diagram.csv"
    frame = pd.read_csv(path)
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    names = {
        "turnkey_soliton_gain_above_loss": "直接开机就能有孤子",
        "extended_soliton_gain_below_loss": "超出开机区之后的孤子",
    }
    for region, part in frame.groupby("region"):
        part = part.sort_values("normalized_mode_shift_Delta")
        ax.plot(part["normalized_mode_shift_Delta"], part["normalized_comb_detuning_zeta"], label=names.get(region, region))
    ax.set_xlabel("归一化模式偏移")
    ax.set_ylabel("归一化梳失谐")
    ax.set_title("公开相图：孤子存在的边界")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS / "public_phase_boundaries.png", dpi=160)
    plt.close(fig)


def shape_plot(labeled: pd.DataFrame) -> None:
    colors = {"narrow": "#9D755D", "smooth": "#54A24B", "modulated": "#E45756"}
    names = {"narrow": "谱还没展开", "smooth": "平滑包络", "modulated": "包络起伏"}
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    for shape, part in labeled.groupby("shape"):
        ax.scatter(part["mu_span_20db"], part["sech_rms"], s=14, alpha=0.75, c=colors[shape], label=names[shape])
    ax.axvline(SPAN_MIN, color="#333333", lw=0.8, ls="--")
    ax.axhline(RMS_MAX, color="#333333", lw=0.8, ls="--")
    ax.set_xlabel("20 dB 模式跨度")
    ax.set_ylabel("相对 sech$^2$ 包络的残差")
    ax.set_title("公开光谱的三种形态")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(RESULTS / "public_states_span_rms.png", dpi=160)
    plt.close(fig)


def user_spectra() -> pd.DataFrame:
    folder = ROOT / "data" / "homework_spark"
    rows = []
    for name in ("spectra_unlocked.csv", "spectra_cw_locked.csv", "spectra_platicon.csv", "spectra_mi_legacy.csv"):
        frame = pd.read_csv(folder / name)
        values = []
        for item in frame["dB_norm"]:
            text = str(item).replace("i", "j")
            try:
                values.append(complex(text).real)
            except ValueError:
                values.append(float(item))
        series = np.asarray(values, dtype=float)
        rows.append({"file": name, "db_span": float(series.max() - series.min()), "db_std": float(series.std())})
    table = pd.DataFrame(rows)
    table.to_csv(RESULTS / "user_spectra_shape.csv", index=False)
    print(table.to_string(index=False))
    return table


def main() -> None:
    os.environ.setdefault("JAVA_HOME", r"C:\Program Files\Eclipse Adoptium\jdk-21.0.4.7-hotspot")
    labeled = assign_shape(pd.read_csv(FEATURES))
    labeled.to_csv(RESULTS / "public_spectra_states.csv", index=False, encoding="utf-8-sig")
    print(labeled.groupby(["split", "shape"]).size())
    shape_plot(labeled)
    phase_plot()
    user_spectra()
    spark = (
        SparkSession.builder.appName("public-comb")
        .master("local[2]")
        .config("spark.driver.memory", "2g")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    report = {
        "shape_rule": {"narrow_if_span_below": SPAN_MIN, "smooth_if_sech_rms_below": RMS_MAX},
        "classification": classify(spark, labeled),
        "heater": heater_fit(spark),
    }
    hybridization(spark)
    spark.stop()
    (RESULTS / "public_spark_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
