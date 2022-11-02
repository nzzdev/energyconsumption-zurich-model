import math
import pandas as pd
from pathlib import Path
from prophet.serialize import model_from_json

DATA_PATH_2022 = Path('./data/consumption/2022.parquet')

df_2022 = pd.read_parquet(DATA_PATH_2022, engine='pyarrow')
df_2022['Date'] = pd.to_datetime(df_2022['Date'])
df_2022['y'] = df_2022['NE5Consumption'] + df_2022['NE7Consumption']

df_2022['CosYearTemp'] = df_2022.apply(lambda row: row['Temperature'] * math.cos(row['Date'].dayofyear * 2 * math.pi / 365), axis=1)
df_2022['SinYearTemp'] = df_2022.apply(lambda row: row['Temperature'] * math.sin(row['Date'].dayofyear * 2 * math.pi / 365), axis=1)

df_2022.rename(columns={'Date': 'ds'}, inplace=True)
df_2022 = df_2022.sort_values('ds')

df_2022 = df_2022.rolling(7, on='ds').mean().dropna()

# Load Modell
with open('./data/model/totalconsumption_rolling7day.json', 'r') as fin:
    m = model_from_json(fin.read())  # Load model
forecastTrain = m.predict(df_2022)
print(forecastTrain.tail())