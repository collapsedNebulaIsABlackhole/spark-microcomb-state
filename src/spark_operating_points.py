"""Spark MLlib：用失谐和扫描方向预测状态，并检验单暗脉冲支路是不是直线。"""

import sys
from collections import Counter

from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import StandardScaler, StringIndexer, VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit, when


def main() -> None:
    csv_path = sys.argv[1]
    spark = (
        SparkSession.builder.appName("comb-operating-points")
        .master("local[2]")
        .config("spark.driver.memory", "1g")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    frame = (
        spark.read.option("header", True)
        .option("inferSchema", True)
        .csv(csv_path)
    )
    frame = frame.withColumn(
        "rev_flag",
        when(col("detuning_mode") == lit("rev_memory"), lit(1.0)).otherwise(lit(0.0)),
    )
    indexed = StringIndexer(inputCol="state", outputCol="label", handleInvalid="keep").fit(frame).transform(frame)
    assembled = VectorAssembler(inputCols=["alpha_L", "rev_flag"], outputCol="raw_features").transform(indexed)
    train_raw, test_raw = assembled.randomSplit([0.7, 0.3], seed=7)
    scaler = StandardScaler(inputCol="raw_features", outputCol="features", withMean=True, withStd=True).fit(train_raw)
    train = scaler.transform(train_raw)
    test = scaler.transform(test_raw)
    model = LogisticRegression(featuresCol="features", labelCol="label", family="multinomial", maxIter=100).fit(train)
    predicted = model.transform(test)
    pairs = predicted.select("label", "prediction").collect()
    correct = sum(row.label == row.prediction for row in pairs)
    accuracy = correct / len(pairs)
    labels = sorted({row.label for row in pairs})
    f1_values = []
    for label in labels:
        tp = sum(row.label == label and row.prediction == label for row in pairs)
        fp = sum(row.label != label and row.prediction == label for row in pairs)
        fn = sum(row.label == label and row.prediction != label for row in pairs)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1_values.append(0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall))
    macro_f1 = sum(f1_values) / len(f1_values)
    print("SPARK_LABEL_COUNTS", dict(Counter(row.label for row in pairs)))

    branch = frame.filter((col("state") == lit("platicon")) & (col("detuning_mode") == lit("rev_memory")))
    branch_vec = VectorAssembler(inputCols=["alpha_L"], outputCol="features").transform(branch)
    reg = LinearRegression(featuresCol="features", labelCol="Pep", maxIter=50).fit(branch_vec)
    summary = reg.summary

    print("SPARK_ROWS", frame.count())
    print("SPARK_TRAIN", train.count())
    print("SPARK_TEST", test.count())
    print("SPARK_LOGISTIC_ACCURACY", round(accuracy, 4))
    print("SPARK_LOGISTIC_MACRO_F1", round(macro_f1, 4))
    print("SPARK_PEP_INTERCEPT", round(reg.intercept, 6))
    print("SPARK_PEP_SLOPE", round(float(reg.coefficients[0]), 6))
    print("SPARK_PEP_R2", round(summary.r2, 4))
    print("SPARK_PEP_RMSE", round(summary.rootMeanSquaredError, 4))
    spark.stop()


if __name__ == "__main__":
    main()
