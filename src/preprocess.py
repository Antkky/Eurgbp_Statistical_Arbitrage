import pandas as pd
import numpy as np
import os
from logger import log
import statsmodels.api as sm
from ta.momentum import RSIIndicator
from ta.trend import MACD

def Process_Test_Data(datapath1: str, datapath2: str, debug: bool = False):
    filename1, filename2 = os.path.split(datapath1)[1], os.path.split(datapath2)[1]
    symbolname1, symbolname2 = filename1.split(".", 1)[0], filename2.split(".", 1)[0]
    data_list = {}

    try:
        data_list[symbolname1] = pd.read_csv(datapath1, index_col="Gmt time")
        data_list[symbolname2] = pd.read_csv(datapath2, index_col="Gmt time")

        data_list[symbolname1].index = pd.to_datetime(data_list[symbolname1].index, format="%d.%m.%Y %H:%M:%S.%f")
        data_list[symbolname2].index = pd.to_datetime(data_list[symbolname2].index, format="%d.%m.%Y %H:%M:%S.%f")

        if debug:
            log(f"Data Successfully Loaded", "preprocess.py", "process_data.py")
    except Exception as e:
        log(f"Data Failed to Load\n{e}", "preprocess.py", "process_data.py")
        return

    # Align both datasets to the same index
    data_list[symbolname1] = data_list[symbolname1].dropna(inplace=True)
    data_list[symbolname2] = data_list[symbolname2].dropna(inplace=True)
    common_index = data_list[symbolname1].index.intersection(data_list[symbolname2].index)
    data_list[symbolname1] = data_list[symbolname1].reindex(common_index)
    data_list[symbolname2] = data_list[symbolname2].reindex(common_index)

    return data_list[symbolname1], data_list[symbolname2]

def rolling_cointegration_score(asset1_series, asset2_series, window=60):
    scores = []
    for i in range(len(asset1_series)):
        if i < window:
            scores.append(np.nan)
        else:
            y = asset1_series[i - window:i]
            x = asset2_series[i - window:i]
            x = sm.add_constant(x)
            model = sm.OLS(y, x).fit()
            residuals = model.resid
            score = sm.tsa.adfuller(residuals)[0]
            scores.append(score)
    return scores

def Process_Train_Data(asset1: pd.DataFrame, asset2: pd.DataFrame) -> pd.DataFrame:
    correlation_window_size = 30
    spread_window_size = 30

    df = pd.DataFrame()

    # Ensure the indexes match
    df['asset1'] = asset1["Close"]
    df['asset2'] = asset2.values["Close"]

    # Rolling Spread-ZScore
    spread = df['asset1'] - df['asset2']
    rolling_mean = spread.rolling(spread_window_size).mean()
    rolling_std = spread.rolling(spread_window_size).std()
    df['Rolling Spread-ZScore'] = (spread - rolling_mean) / rolling_std

    # Rolling Correlation
    df['Rolling Correlation'] = df['asset1'].rolling(correlation_window_size).corr(df['asset2'])

    # Rolling Cointegration Score
    df['Rolling Cointegration Score'] = rolling_cointegration_score(df['asset1'], df['asset2'])

    # Relative Strength Score (Ratio)
    df['Relative Strength Score'] = df['asset1'] / df['asset2']

    # Relative Strength Index (RSI)
    df['Relative Strength Index'] = RSIIndicator(close=df['asset1'], window=14).rsi()

    # Moving Average Convergence Divergence (MACD)
    macd = MACD(close=df['asset1'], window_slow=26, window_fast=12, window_sign=9)
    df['MACD'] = macd.macd()

    df = df.dropna()
    df.to_csv("./data/processed/train_data.csv")
    return df
