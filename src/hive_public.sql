CREATE DATABASE IF NOT EXISTS comb;

DROP TABLE IF EXISTS comb.states;
DROP TABLE IF EXISTS comb.step_no;
DROP TABLE IF EXISTS comb.step_partial;
DROP TABLE IF EXISTS comb.step_complete;

CREATE EXTERNAL TABLE comb.states (
  split STRING,
  file_name STRING,
  n_lines INT,
  lines_within_20db INT,
  mu_span_20db DOUBLE,
  peak_power_dbm DOUBLE,
  mu_centroid DOUBLE,
  sech_rms DOUBLE,
  shape STRING
)
ROW FORMAT DELIMITED
FIELDS TERMINATED BY ','
STORED AS TEXTFILE
LOCATION '/user/bigdata/public/states'
TBLPROPERTIES ('skip.header.line.count'='1');

CREATE EXTERNAL TABLE comb.step_no (
  scan_current_ma DOUBLE,
  power_bin DOUBLE,
  hit_count BIGINT
)
ROW FORMAT DELIMITED
FIELDS TERMINATED BY ','
STORED AS TEXTFILE
LOCATION '/user/bigdata/public/step_no'
TBLPROPERTIES ('skip.header.line.count'='1');

CREATE EXTERNAL TABLE comb.step_partial (
  scan_current_ma DOUBLE,
  power_bin DOUBLE,
  hit_count BIGINT
)
ROW FORMAT DELIMITED
FIELDS TERMINATED BY ','
STORED AS TEXTFILE
LOCATION '/user/bigdata/public/step_partial'
TBLPROPERTIES ('skip.header.line.count'='1');

CREATE EXTERNAL TABLE comb.step_complete (
  scan_current_ma DOUBLE,
  power_bin DOUBLE,
  hit_count BIGINT
)
ROW FORMAT DELIMITED
FIELDS TERMINATED BY ','
STORED AS TEXTFILE
LOCATION '/user/bigdata/public/step_complete'
TBLPROPERTIES ('skip.header.line.count'='1');

SELECT shape, split, COUNT(*) AS n,
       ROUND(AVG(peak_power_dbm), 2) AS peak_avg,
       ROUND(AVG(mu_span_20db), 1) AS span_avg
FROM comb.states
WHERE shape IN ('smooth', 'modulated', 'narrow')
GROUP BY shape, split
ORDER BY shape, split;

SELECT condition,
       SUM(hit_count) AS hits,
       ROUND(SUM(CASE WHEN power_bin < 0.15 THEN hit_count ELSE 0 END) / SUM(hit_count), 4) AS low_share,
       ROUND(SUM(CASE WHEN power_bin >= 0.15 AND power_bin < 0.65 THEN hit_count ELSE 0 END) / SUM(hit_count), 4) AS mid_share,
       ROUND(SUM(CASE WHEN power_bin >= 0.65 THEN hit_count ELSE 0 END) / SUM(hit_count), 4) AS high_share
FROM (
  SELECT 'no' AS condition, power_bin, hit_count FROM comb.step_no WHERE hit_count IS NOT NULL
  UNION ALL
  SELECT 'partial', power_bin, hit_count FROM comb.step_partial WHERE hit_count IS NOT NULL
  UNION ALL
  SELECT 'complete', power_bin, hit_count FROM comb.step_complete WHERE hit_count IS NOT NULL
) scans
GROUP BY condition
ORDER BY condition;
