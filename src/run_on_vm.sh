#!/usr/bin/env bash
set -eu

docker cp "$HOME/comb-job/public/pyspark-bundle.tgz" master:/tmp/pyspark-bundle.tgz
docker exec master tar -xzf /tmp/pyspark-bundle.tgz -C /opt
docker exec master mkdir -p /tmp/public
for name in states.csv heater.csv step_no.csv step_partial.csv step_complete.csv spark_public_vm.py; do
  docker cp "$HOME/comb-job/public/$name" "master:/tmp/public/$name"
done
docker exec master hdfs dfs -mkdir -p /user/bigdata/public
docker exec master hdfs dfs -put -f \
  /tmp/public/states.csv \
  /tmp/public/heater.csv \
  /tmp/public/step_no.csv \
  /tmp/public/step_partial.csv \
  /tmp/public/step_complete.csv \
  /user/bigdata/public/
docker exec master hdfs dfs -ls -h /user/bigdata/public
docker exec \
  -e PYTHONPATH=/opt \
  -e HADOOP_CONF_DIR=/usr/local/hadoop/etc/hadoop \
  -e PYSPARK_PYTHON=python3 \
  -e SPARK_HOME=/opt/pyspark \
  master \
  bash -lc '/opt/pyspark/bin/spark-submit --master local[2] --driver-memory 1g /tmp/public/spark_public_vm.py hdfs://master:9000/user/bigdata/public'
