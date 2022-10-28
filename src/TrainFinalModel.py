# Databricks notebook source
# MAGIC %md # Final Model
# MAGIC Prophet is a model from Facebook for time series forecasting. It combines piecewise linear(ish) functions with fourier series for cyclical properties and categorical encodings of holidays and events. See more here https://facebook.github.io/prophet/docs/quick_start.html#python-api\

# COMMAND ----------

# MAGIC %pip install prophet

# COMMAND ----------

import math
from datetime import datetime
from datetime import timedelta
from pyspark.sql import functions as f
from pyspark.sql.types import *
import pandas as pd
from prophet import Prophet
from prophet.plot import add_changepoints_to_plot
from prophet.utilities import regressor_coefficients
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from matplotlib import pyplot as plt
from prophet.serialize import model_to_json, model_from_json

# COMMAND ----------

# MAGIC %md ## Prepare Data

# COMMAND ----------

# https://en.wikipedia.org/wiki/COVID-19_pandemic_in_Switzerland
COVID_START = datetime(2020, 3, 1).date()
# https://www.admin.ch/gov/en/start/documentation/media-releases.msg-id-84127.html
# date taken as the second easing of restrictions during 2021
COVID_END = datetime(2021, 6, 1).date()

MODEL_PATH = "/ConsumptionModel/Models/"
TRAIN_DATA_PATH = "/ConsumptionModel/Data/final_train.snappy.parquet"

MODEL_NAME = "totalconsumption_rolling7day.json"
CONSUMPTION_EXPR = "NE5Consumption + NE7Consumption" # i.e. "NE5Consumption + NE7Consumption", "NE5Consumption", "NE7Consumption"
ROLLING_WINDOW = 7
CONFIDENCE_INTERVAL = 0.95
USE_WEEK_SEASONALITY = False
USE_HOLIDAYS = False

# COMMAND ----------

def loadParquetToPandas(path):
  return (spark.read.format("parquet").load(path)
                                      .withColumn("y", f.expr(CONSUMPTION_EXPR))
                                      .withColumn("CosYearTemp", f.expr("Temperature * cos(dayofyear(Date) * 2 * pi() / 365)"))
                                      .withColumn("SinYearTemp", f.expr("Temperature * sin(dayofyear(Date) * 2 * pi() / 365)"))
                                      .selectExpr("Date as ds", "y", "CosYearTemp", "SinYearTemp")
                                      .orderBy("ds")
         ).toPandas().rolling(ROLLING_WINDOW, on="ds").mean().dropna()[:-ROLLING_WINDOW]

train = loadParquetToPandas(TRAIN_DATA_PATH)

# COMMAND ----------

# MAGIC %md ## Build Model

# COMMAND ----------

# add covid as a one-off holiday
covid = pd.DataFrame([
    {'holiday': 'covid', 'ds': COVID_START, 'lower_window': 0, 'ds_upper': COVID_END}
])

covid['upper_window'] = (covid['ds_upper'] - covid['ds']).dt.days

# COMMAND ----------

m = Prophet(holidays = covid, weekly_seasonality = USE_WEEK_SEASONALITY, changepoint_prior_scale = 0.005, changepoint_range=1, interval_width=CONFIDENCE_INTERVAL, uncertainty_samples = 10000)

if USE_HOLIDAYS:
  m.add_country_holidays(country_name='CH')
  
m.add_regressor("CosYearTemp", prior_scale = 0.05, standardize = True, mode = "additive")
m.add_regressor("SinYearTemp", prior_scale = 0.05, standardize = True, mode = "additive")
m.fit(train)

# COMMAND ----------

forecastTrain = m.predict(train)

# COMMAND ----------

fig = m.plot_components(forecastTrain)

# COMMAND ----------

fig = m.plot(forecastTrain)
a = add_changepoints_to_plot(fig.gca(), m, forecastTrain)

# COMMAND ----------

# MAGIC %md ## Error Metrics Train

# COMMAND ----------

mse = mean_squared_error(train["y"].to_numpy(), forecastTrain["yhat"].to_numpy())
mse

# COMMAND ----------

rmse = math.sqrt(mse)
rmse

# COMMAND ----------

mean_absolute_percentage_error(train["y"].to_numpy(), forecastTrain["yhat"].to_numpy())

# COMMAND ----------

regressor_coefficients(m)

# COMMAND ----------

# MAGIC %md ## Plots Train

# COMMAND ----------

plotData = pd.DataFrame(
  {
    "Actual": train["y"].array,
    "PredictedMean": forecastTrain["yhat"].array,
    "PredictedHigh": forecastTrain["yhat_upper"].array,
    "PredictedLow": forecastTrain["yhat_lower"].array
  },
  index=train["ds"]
)

plotData.index = pd.to_datetime(plotData.index)
plotData = plotData.sort_index()

# COMMAND ----------

def plot(data):
  fig, ax = plt.subplots()
  fig.set_size_inches(12, 8)
  fig.patch.set_facecolor('white')
  ax.plot(data.index, data["Actual"])
  ax.fill_between(data.index, data["PredictedHigh"], data["PredictedLow"], color='orange', alpha=.2)

# COMMAND ----------

plot(plotData)

# COMMAND ----------

plot(plotData[(plotData.index >= datetime(2019, 1, 1)) & (plotData.index < datetime(2019, 3, 1))])

# COMMAND ----------

# MAGIC %md ## Save Model

# COMMAND ----------

with open(MODEL_PATH + MODEL_NAME, 'w') as fout:
    fout.write(model_to_json(m))
    
