import math
import itertools
import pandas as pd
import numpy as np
import statsmodels.api as sm


def pdq_combinations(season_period, max_p, max_d, max_q):
    p, d, q = range(0, max_p+1), range(0, max_d+1), range(0, max_q+1)
    pdq = list(itertools.product(p, d, q))
    seasonal_pdq = [(x[0], x[1], x[2], season_period) for x in pdq]
    return pdq, seasonal_pdq

def pdq_hyperparam_search(df, season_period, max_p=1, max_d=1, max_q=1):
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