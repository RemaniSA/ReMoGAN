# file: data_handler.py
# !/usr/bin/env python3

# -*- coding: utf-8 -*-
'''
Copyright (c) 2025 Shaan Ali Remani. All rights reserved.
Licensed under the MIT License.
'''

import pandas as pd
import numpy as np

def compute_log_prices(ticker, date_col='Date', price_col='Close'):
    """
    Loads time series data, sets a datetime index, and computes log-prices.

    Args:
        ticker (str): The ticker symbol, used to find the CSV file (e.g., 'SPY').
        date_col (str): The name of the date column in the CSV.
        price_col (str): The name of the price column (e.g., 'Close' or 'Adj Close').

    Returns:
        pd.DataFrame: A DataFrame with log prices indexed by date.
    """
    # Construct file path and read data
    file_path = f'data/{ticker}.csv'
    df = pd.read_csv(file_path)
    
    # Convert date column to datetime objects and set as index
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col)
    
    # Compute log prices
    log_prices = df[[price_col]].apply(np.log)
    
    # Drop any missing values that might result from the log transformation
    log_prices = log_prices.dropna()
    
    print(f"Loaded and prepared data for {ticker}. Date range: {log_prices.index.min().strftime('%Y-%m-%d')} to {log_prices.index.max().strftime('%Y-%m-%d')}")
 
    return log_prices