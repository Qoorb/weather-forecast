import pandas as pd
import data_utils as du
import models as mdl


dataset = pd.read_csv("./data/weather_data_1940_2024.csv", sep=',')
dataset.rename(columns = {'temperature_2m_mean (°C)':'temperature'}, inplace = True)

dataset = du.extract_features_from_date(dataset, 'time')

data = du.prepare_data(dataset, 'temperature')

setup = mdl.Models(data)

setup.compare()

setup.forecast()
