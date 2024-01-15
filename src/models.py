from datetime import date, timedelta

import pandas as pd

import data_utils as du

from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

from catboost import CatBoostRegressor


class Models:

    def __init__(self, data: list) -> None:
        self.models = [SVR(), RandomForestRegressor(), LinearRegression(), CatBoostRegressor()]

        self.X_train = data[0]
        self.X_valid = data[1]
        self.Y_train = data[2]
        self.Y_valid = data[3]

    def compare(self) -> None:

        model_metrics = {
            # model_name: metrics
        }

        for model in self.models:
            metrics = []

            model.fit(self.X_train, self.Y_train)

            Y_pred = model.predict(self.X_valid)

            metrics.append(mean_absolute_error(self.Y_valid, Y_pred))
            metrics.append(r2_score(self.Y_valid, Y_pred))

            model_metrics[str(model)] = metrics
        
        for model_name, metrics in model_metrics.items():
            print(f"{model_name}: \nMAE: {metrics[0]} \nR2: {metrics[1]}")
    
    def forecast(self) -> None:

        model_forecasts = {
            # model_name: forecast
        }
        cols = ['time']

        start_day = date.today()
        end_day = start_day + timedelta(days = 6)
        
        values = du.dates_generator(start_day, end_day)

        data_unseen = pd.DataFrame(data=values, columns = cols)
        data_unseen['time'] = pd.to_datetime(data_unseen['time'])

        data_unseen = du.extract_features_from_date(data_unseen, 'time')

        for model in self.models:
            model_forecasts[str(model)] = model.predict(data_unseen)
        
        for model_name, forecast in model_forecasts.items():
            print(f"{model_name}: \nweekly weather forecast: {forecast}")
