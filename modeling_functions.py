import math
import itertools
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.base import BaseEstimator, TransformerMixin
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM


class LSTM_Reshaper(BaseEstimator, TransformerMixin):

    def __init__(self, variables=None):
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        return X.reshape(X.shape[0], 1, X.shape[1])


def keras_model_wrapper():
    model = Sequential()
    model.add(LSTM(4, batch_input_shape=(X.shape[0], 1, X.shape[1]), stateful=True))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

def diff_handle_missing(df, diff, lag=0):
    return df.diff(diff).iloc[diff + lag:]

def ts_to_supervised(df, lag=1):
    df = pd.DataFrame(df)
    idx, shift = df.index, df.index.shift(-lag)
    new_idx = idx.union(shift)
    return df.reindex(new_idx).shift(-lag, fill_value=0).iloc[lag:-lag]

def preprocess_lstm(df, lag=1, diff=1):
    X = diff_handle_missing(df, diff)
    y = ts_to_supervised(X, lag)
    X = X[:-lag]
#     y = ts_to_supervised(df, lag)
#     X, y = diff_handle_missing(df, diff), diff_handle_missing(y, diff)
    return X, y

def pdq_combinations(season_period, max_p, max_d, max_q):
    p, d, q = range(0, max_p+1), range(0, max_d+1), range(0, max_q+1)
    pdq = list(itertools.product(p, d, q))
    seasonal_pdq = [(x[0], x[1], x[2], season_period) for x in pdq]
    return pdq, seasonal_pdq

def pdq_hyperparam_search(df, season_period, max_p=1, max_d=1, max_q=1):
    """Generates and searches a list of pdq and season values to find best hyperparameters."""

    pdq, seasonal_pdq = pdq_combinations(season_period, max_p, max_d, max_q)
    all_results=[]
    for param in pdq:
        for param_seasonal in seasonal_pdq:
            try:
                mod = sm.tsa.statespace.SARIMAX(df, order=param, seasonal_order=param_seasonal, enforce_stationarity=False, enforce_invertibility=False)
                results = mod.fit()
                rmse = math.sqrt(results.mse)
                print('ARIMA{}x{}12 - rmse:{}'.format(param, param_seasonal, rmse))
                all_results.append([param, param_seasonal, rmse])
            except:
                continue
    return pd.DataFrame(all_results, columns = ["PDQ", "S-PDQ", "Error"]).sort_values("Error", ascending=True).reset_index(drop=True)

def pdq_hyperparam_df_search(df, pdq_params, new_period=None, n_combs=5):
    """Like pdq_hyperparam search, but iterates through a specific amount of values in a DataFrame. Used to check values if initial data needed resampling
    for speed."""

    all_results=[]
    for i in range(0, n_combs):
        param, param_seasonal = pdq_params.iloc[i:i+1, 0].to_list()[0], pdq_params.iloc[i:i+1, 1].to_list()[0]
        if new_period:
            param_seasonal = list(param_seasonal[0:3])+[new_period]
        try:
            mod = sm.tsa.statespace.SARIMAX(df, order=param, seasonal_order=param_seasonal, enforce_stationarity=False, enforce_invertibility=False)
            results = mod.fit()
            rmse = math.sqrt(results.mse)
            print('ARIMA{}x{}12 - rmse:{}'.format(param, param_seasonal, rmse))
            all_results.append([param, param_seasonal, rmse])
        except:
            continue
    return pd.DataFrame(all_results, columns = ["PDQ", "S-PDQ", "Error"]).sort_values("Error", ascending=True).reset_index(drop=True)