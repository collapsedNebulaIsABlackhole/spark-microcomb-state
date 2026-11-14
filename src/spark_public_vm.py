"""在虚拟机里用 Spark 跑公开数据。只依赖 pyspark，不画图。

用法：spark-submit spark_public_vm.py /数据目录
目录里要有 states.csv、heater.csv，以及三种耦合的 step_count csv。
"""

from __future__ import annotations

import sys

from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import StandardScaler, StringIndexer, VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, input_file_name, lit, sum as spark_sum, when


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


def classify(spark, path):
    frame = spark.read.option("header", True).option("inferSchema", True).csv(path + "/states.csv")
    frame = frame.filter(col("shape").isin("smooth", "modulated"))
    indexed = StringIndexer(inputCol="shape", outputCol="label").fit(frame).transform(frame)

    def one_model(columns):
        assembled = VectorAssembler(inputCols=columns, outputCol="raw_features").transform(indexed)
        train_raw = assembled.filter(col("split") == lit("TrainingSet"))
        scaler = StandardScaler(inputCol="raw_features", outputCol="features", withMean=True, withStd=True).fit(train_raw)
        model = LogisticRegression(featuresCol="features", labelCol="label", maxIter=100).fit(scaler.transform(train_raw))
        for name in ("TestingSet", "ValidationSet"):
            rows = model.transform(scaler.transform(assembled.filter(col("split") == lit(name)))).select("label", "prediction").collect()
            pairs = [(row.label, row.prediction) for row in rows]
            correct = sum(got == pred for got, pred in pairs)
            accuracy = correct / len(pairs) if pairs else 0.0
            print("VM_CLASS", ",".join(columns), name, len(pairs), round(accuracy, 4), round(macro_f1(pairs), 4))

    print("VM_ROWS", frame.count())
    one_model(["lines_within_20db", "mu_span_20db", "peak_power_dbm", "mu_centroid"])
    one_model(["lines_within_20db", "mu_span_20db", "mu_centroid"])


def heater_fit(spark, path):
    frame = spark.read.option("header", True).option("inferSchema", True).csv(path + "/heater.csv")
    assembled = VectorAssembler(inputCols=["AR_heater_power_mW"], outputCol="features").transform(frame)
    bandwidth = LinearRegression(featuresCol="features", labelCol="spectral_envelope_bandwidth_3dB_nm", maxIter=50).fit(assembled)
    linewidth = LinearRegression(featuresCol="features", labelCol="lorentzian_linewidth_Hz", maxIter=50).fit(assembled)
    print("VM_HEATER_N", frame.count())
    print("VM_BANDWIDTH_SLOPE", round(float(bandwidth.coefficients[0]), 6))
    print("VM_BANDWIDTH_R2", round(float(bandwidth.summary.r2), 4))
    print("VM_LINEWIDTH_SLOPE", round(float(linewidth.coefficients[0]), 6))
    print("VM_LINEWIDTH_R2", round(float(linewidth.summary.r2), 4))


def hybridization(spark, path):
    frame = spark.read.option("header", True).option("inferSchema", True).csv(path + "/step_*.csv")
    frame = frame.withColumn(
        "condition",
        when(input_file_name().contains("partial"), lit("partial"))
        .when(input_file_name().contains("complete"), lit("complete"))
        .otherwise(lit("no")),
    )
    power = col("normalized_comb_power_bin_center")
    hits = col("hit_count")
    table = frame.groupBy("condition").agg(
        spark_sum(hits).alias("hits"),
        spark_sum(when(power < 0.15, hits).otherwise(0)).alias("low"),
        spark_sum(when((power >= 0.15) & (power < 0.65), hits).otherwise(0)).alias("mid"),
        spark_sum(when(power >= 0.65, hits).otherwise(0)).alias("high"),
    ).collect()
    for row in table:
        total = row.hits or 1
        print(
            "VM_HYBRID",
            row.condition,
            int(row.hits),
            round(row.low / total, 4),
            round(row.mid / total, 4),
            round(row.high / total, 4),
        )


def main():
    data = sys.argv[1]
    spark = (
        SparkSession.builder.appName("public-comb-vm")
        .config("spark.driver.memory", "1g")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    classify(spark, data)
    heater_fit(spark, data)
    hybridization(spark, data)
    spark.stop()
    print("VM_DONE")


if __name__ == "__main__":
    main()
