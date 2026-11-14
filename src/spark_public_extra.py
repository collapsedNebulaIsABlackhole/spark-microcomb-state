"""虚拟机上的补充：Spark SQL 分组、K-means、决策树。读 HDFS。"""

from __future__ import annotations

import sys

from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import StandardScaler, StringIndexer, VectorAssembler
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit


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
        SparkSession.builder.appName("public-comb-extra")
        .config("spark.driver.memory", "512m")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    states = spark.read.option("header", True).option("inferSchema", True).csv(path + "/states")
    print("SQL_ROWS", states.count())
    counts = states.groupBy("shape", "split").count().orderBy("shape", "split").collect()
    for row in counts:
        print("SQL_COUNT", row["shape"], row["split"], row["count"])

    feature_cols = ["lines_within_20db", "mu_span_20db", "peak_power_dbm", "mu_centroid"]
    assembled = VectorAssembler(inputCols=feature_cols, outputCol="raw_features").transform(states)
    scaler = StandardScaler(inputCol="raw_features", outputCol="features", withMean=True, withStd=True).fit(assembled)
    scaled = scaler.transform(assembled)
    model = KMeans(k=3, seed=7, featuresCol="features").fit(scaled)
    predicted = model.transform(scaled)
    pairs = predicted.groupBy("shape", "prediction").count().orderBy("shape", "prediction").collect()
    print("KMEANS_K", 3)
    for row in pairs:
        print("KMEANS_CELL", row["shape"], int(row["prediction"]), row["count"])
    for center in model.clusterCenters():
        print("KMEANS_CENTER", " ".join(f"{value:.3f}" for value in center))

    usable = scaled.filter(col("shape").isin("smooth", "modulated"))
    indexed = StringIndexer(inputCol="shape", outputCol="label").fit(usable).transform(usable)
    train = indexed.filter(col("split") == lit("TrainingSet"))
    tree = DecisionTreeClassifier(featuresCol="features", labelCol="label", maxDepth=4, seed=7).fit(train)
    for name in ("TestingSet", "ValidationSet"):
        rows = tree.transform(indexed.filter(col("split") == lit(name))).select("label", "prediction").collect()
        got = [(row.label, row.prediction) for row in rows]
        correct = sum(label == pred for label, pred in got)
        accuracy = correct / len(got) if got else 0.0
        print("TREE", name, len(got), round(accuracy, 4), round(macro_f1(got), 4))
    spark.stop()
    print("EXTRA_DONE")


if __name__ == "__main__":
    main()
