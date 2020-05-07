import math
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pylab as plt
from statsmodels.tsa.stattools import adfuller


def dickey_fuller_df(df):
    test = adfuller(df)
    dfoutput = pd.Series(test[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in test[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    return dfoutput


def rolling_ts_plot(df, halflife, figsize=(15, 10)):
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)
    mean = df.ewm(halflife=halflife).mean()
    std = df.ewm(halflife=halflife).std()
    df.plot(ax=ax)
    mean.plot(ax=ax)
    std.plot(ax=ax)
    ax.legend(["Original", "Exponentially Weighted Rolling Mean", "Exponentially Weighted Rolling Standard Deviation"]);
    plt.show()


def stationarity_check(df, halflife, figsize=(15, 10)):
    rolling_ts_plot(df, halflife, figsize=figsize)
    return dickey_fuller_df(df)


def plot_sarimax_one_step(model_results, observations, pred_date, date_trim=None, inv_func=None):
    if not date_trim:
        date_trim = train.index[0]
    pred = model_results.get_prediction(start=pd.to_datetime(pred_date), dynamic=False)
    if inv_func:
        pred_ci = inv_func(pred.conf_int())
        pred_mean = inv_func(pred.predicted_mean)
        observations = inv_func(observations)
    else:
        pred_ci = pred.conf_int()
        pred_mean = pred.predicted_mean
    ax = observations[date_trim:].plot(label='observed')
    pred_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7, figsize=(14, 7))
    ax.fill_between(pred_ci.index,
                    pred_ci.iloc[:, 0],
                    pred_ci.iloc[:, 1], color='k', alpha=.2)
    ax.set_xlabel('Date')
    ax.set_ylabel('Furniture Sales')
    plt.legend()
    plt.show()

    actual = observations[pred_date:]
    mse = ((pred_mean - actual) ** 2).mean()
    print(f'Mean Squared Error - {mse}')
    print(f'Root Mean Squared Error - {math.sqrt(mse)}')