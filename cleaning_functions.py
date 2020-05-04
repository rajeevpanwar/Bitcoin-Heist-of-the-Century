import numpy as np
import pandas as pd

def decomma_string(x):
    if type(x) == str:
        x = x.replace(",", "")
    return x

def df_multimap(df, columns, func):
    for column in columns:
        df[column]=df[column].map(func)
    return df

def coinmarketcap_destring(df):
    """Function to convert specific datatypes in the scraped data. Currently replaced with a more efficient function."""

    return df.astype({"Open":float, "High":float, "Low":float, "Close":float, "Volume":int, "Market Cap":int})

def asset_ts_clean(df, format=None):
    df["Date"] = pd.to_datetime(df["Date"], format=format)
    df = df.set_index("Date")
    df = df.applymap(decomma_string)
    df = df.apply(pd.to_numeric, **{"errors":'coerce'})
    return df