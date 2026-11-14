"""逐齿功率关系。不使用绝对峰功率，也不使用相位。"""

from __future__ import annotations

import sys

from pyspark.ml.classification import DecisionTreeClassifier, LogisticRegression
from pyspark.ml.feature import StandardScaler, StringIndexer, VectorAssembler
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit, mean, stddev


FEATURES = ["adj_db", "left_share", "even_share", "gap_frac", "pump_excess_db", "envelope_corr"]


def macro_f1(pairs):
    labels = sorted({label for label, _ in pairs})
    scores = []
    for label in labels:
        tp = sum(got == label and pred == label for got, pred in pairs)
        fp = sum(got != label and pred == label for got, pred in pairs)
        fn = sum(got == label and pred != label for got, pred in pairs)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall))
    return sum(scores) / len(scores) if scores else 0.0


def main():
    path = sys.argv[1]
    spark = (
        SparkSession.builder.appName("tooth-relations")
        .config("spark.driver.memory", "512m")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    frame = spark.read.option("header", True).option("inferSchema", True).csv(path)
    print("TOOTH_ROWS", frame.count())
    stats = frame.groupBy("shape").agg(
        *[mean(name).alias(name + "_mean") for name in FEATURES],
        *[stddev(name).alias(name + "_std") for name in FEATURES],
    ).orderBy("shape")
    for row in stats.collect():
        print("SHAPE", row["shape"])
        for name in FEATURES:
            print("  ", name, round(row[name + "_mean"], 4), round(row[name + "_std"] or 0, 4))

    usable = frame.filter(col("shape").isin("smooth", "modulated")).na.drop(subset=FEATURES)
    assembled = VectorAssembler(inputCols=FEATURES, outputCol="raw_features").transform(usable)
    scaler = StandardScaler(inputCol="raw_features", outputCol="features", withMean=True, withStd=True).fit(assembled)
    scaled = scaler.transform(assembled)
    indexed = StringIndexer(inputCol="shape", outputCol="label").fit(scaled).transform(scaled)
    train = indexed.filter(col("split") == lit("TrainingSet"))
    logistic = LogisticRegression(featuresCol="features", labelCol="label", maxIter=50).fit(train)
    tree = DecisionTreeClassifier(featuresCol="features", labelCol="label", maxDepth=4, seed=7).fit(train)
    for model_name, model in (("LOGISTIC", logistic), ("TREE", tree)):
        for split_name in ("TestingSet", "ValidationSet"):
            rows = model.transform(indexed.filter(col("split") == lit(split_name))).select("label", "prediction").collect()
            got = [(row.label, row.prediction) for row in rows]
            accuracy = sum(label == pred for label, pred in got) / len(got) if got else 0.0
            print(model_name, split_name, len(got), round(accuracy, 4), round(macro_f1(got), 4))
    if hasattr(logistic, "coefficients"):
        print("LOGISTIC_COEF", " ".join(f"{FEATURES[i]}={logistic.coefficients[i]:.3f}" for i in range(len(FEATURES))))
    spark.stop()
    print("TOOTH_VM_DONE")


if __name__ == "__main__":
    main()
