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