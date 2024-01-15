from datetime import timedelta, datetime

import pandas as pd

from sklearn.model_selection import train_test_split


def extract_features_from_date(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    
    df[column_name] = pd.to_datetime(df[column_name])

    #TODO: add date format check
    df['ds_year'] = df[column_name].dt.year
    df['ds_month'] = df[column_name].dt.month
    df['ds_day'] = df[column_name].dt.day

    df = df.drop(column_name, axis=1)

    return df

def prepare_data(data: pd.DataFrame, target: str) -> list:

    X = data.drop([target], axis=1)
    Y = data[target]

    return train_test_split(X, Y, train_size=0.8, test_size=0.2, random_state=0)

def dates_generator(start_date: datetime, end_date: datetime) -> list:
    dates = []

    #TODO: add date format and type check
    # start_date = datetime.strptime(start_date, "%Y-%m-%d")
    # end_date = datetime.strptime(end_date, "%Y-%m-%d")

    delta = timedelta(days=1)

    while start_date <= end_date:
        dates.append(start_date.strftime("%Y-%m-%d"))
        start_date += delta
  
    return dates
