# Databricks notebook source
from datetime import datetime
from datetime import timedelta
from pyspark.sql import functions as f
from pyspark.sql.types import *
from pyspark.sql.window import Window

# COMMAND ----------

CONSUMPTION_DATA_PATH = "/ConsumptionModel/Data/"
CONSUMPTION_NE5_FILE = CONSUMPTION_DATA_PATH + "NE5_Export.csv"
CONSUMPTION_NE7_FILE = CONSUMPTION_DATA_PATH + "NE7_Export.csv"

METEO_PATH = "/MeteoSwiss/Measurement/Delta"
METEO_STATION = "REH"
METEO_TEMP_PARAMETER = "tre200h0"

TIME_FORMAT = "dd.MM.yyyy"
WINDOW_DAYS = 1
START_DATE = datetime(2010, 1, 1)
END_DATE = datetime(2022, 1, 1)

# COMMAND ----------

ne5File = spark.read.csv(CONSUMPTION_NE5_FILE, header=True, inferSchema=True, sep=";")
ne7File = spark.read.csv(CONSUMPTION_NE7_FILE, header=True, inferSchema=True, sep=";")

consumption = (ne5File.alias("ne5")
                      .join(ne7File.alias("ne7"), "Date")
                      .withColumn("NE5Consumption", f.expr("ne5.Value"))
                      .withColumn("NE7Consumption", f.expr("ne7.Value"))
                      .withColumn("Date", f.to_date(f.col("Date"), TIME_FORMAT))
                      .filter((f.col("Date") >= START_DATE) & (f.col("Date") < END_DATE))
                      .select("Date", "NE5Consumption", "NE7Consumption")
              )

# COMMAND ----------

meteoData = (spark.read.format("delta")
                         .load("/MeteoSwiss/Measurement/Delta")
                         .filter((f.col("Station") == METEO_STATION) & (f.col("Parameter") == METEO_TEMP_PARAMETER))
                         .withColumn("Date", f.to_date(f.col("TimestampUtc")))
                         .groupBy("Date")
                         .agg(f.avg(f.col("Value")).alias("Temperature"))
            )

# COMMAND ----------

data = consumption.join(meteoData, "Date")

# COMMAND ----------

data.write.parquet("/ConsumptionModel/Data/final_train.parquet")
