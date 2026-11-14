CREATE DATABASE IF NOT EXISTS comb;

DROP TABLE IF EXISTS comb.operating_points;

CREATE EXTERNAL TABLE comb.operating_points (
  source STRING,
  detuning_mode STRING,
  protocol STRING,
  closed_loop INT,
  i_ma DOUBLE,
  phi_over_pi DOUBLE,
  gvd STRING,
  d2_khz DOUBLE,
  kappa_mhz DOUBLE,
  alpha_l DOUBLE,
  linewidths DOUBLE,
  detuning_ghz DOUBLE,
  al DOUBLE,
  pump_f DOUBLE,
  f_th DOUBLE,
  pump_over_fth DOUBLE,
  pep DOUBLE,
  pem DOUBLE,
  t DOUBLE,
  n_peak INT,
  n_dip INT,
  n_dark INT,
  dark_frac DOUBLE,
  contrast DOUBLE,
  state STRING
)
ROW FORMAT DELIMITED
FIELDS TERMINATED BY ','
STORED AS TEXTFILE
LOCATION '/user/bigdata/comb/operating_points';

SELECT state, detuning_mode,
       COUNT(*) AS n,
       ROUND(AVG(pep), 4) AS pep_avg,
       ROUND(MIN(pep), 4) AS pep_min,
       ROUND(MAX(pep), 4) AS pep_max,
       ROUND(AVG(dark_frac), 4) AS dark_avg,
       ROUND(MIN(alpha_l), 2) AS alpha_min,
       ROUND(MAX(alpha_l), 2) AS alpha_max
FROM comb.operating_points
GROUP BY state, detuning_mode
ORDER BY state, detuning_mode;
