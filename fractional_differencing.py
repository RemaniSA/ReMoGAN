# file: fractional_differencing.py
# !/usr/bin/env python3

# -*- coding: utf-8 -*-
'''
Copyright (c) 2025 Shaan Ali Remani. All rights reserved.
Licensed under the MIT License.
'''

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
import data_handler as dh

def get_weights_ffd(d, thres):
    """
    Calculates weights for Fixed-Width Window Fractional Differencing (FFD) method.
    Based on López de Prado, M. (2018).
    """
    w, k = [1.], 1
    while True:
        w_ = -w[-1] / k * (d - k + 1)
        if abs(w_) < thres:
            break
        w.append(w_)
        k += 1
    return np.array(w[::-1]).reshape(-1, 1)

def frac_diff_ffd(series, d, thres=1e-5):
    """
    Applies FFD to a time series.
    Based on López de Prado, M. (2018).
    """
    w = get_weights_ffd(d, thres)
    width = len(w)
    df_ = {}
    for name in series.columns:
        series_f = series[[name]].fillna(method='ffill').dropna()
        df_temp = pd.Series(dtype=float)
        # Apply the weights to the series using a fixed-width window
        for i in range(width - 1, series_f.shape[0]):
            window = series_f.iloc[i - (width - 1) : i + 1]
            if len(window) == width and np.all(np.isfinite(window.values)):
                df_temp[series_f.index[i]] = np.dot(w.T, window.values)[0, 0]
        
        df_[name] = df_temp.copy(deep=True)
    df = pd.concat(df_, axis=1)
    return df

def find_minimum_d(ticker, orders=np.linspace(0, 1, 11)):
    """
    Orchestrates the search for the minimum differencing order 'd' for a given ticker.
    Based on López de Prado, M. (2018)'s methodology.

    Args:
        ticker (str): The ticker symbol.
        orders (np.ndarray): An array of 'd' values to test.

    Returns:
        pd.DataFrame: A DataFrame with the ADF statistic, p-value, and correlation for each 'd'.
    """
    # Use data_handler to load prepared log prices
    log_price_series = dh.compute_log_prices(ticker=ticker)
    
    if log_price_series.empty:
        print(f"Data loading failed for {ticker}. Aborting analysis.")
        return pd.DataFrame()  # Return empty if data loading failed

    results = []
    for d in orders:
        differenced_series = frac_diff_ffd(log_price_series, d)
        common_indices = log_price_series.index.intersection(differenced_series.index)
        
        # Ensure there are common data points to compare
        if len(common_indices) > 1:
            corr = np.corrcoef(log_price_series.loc[common_indices].iloc[:, 0], 
                               differenced_series.loc[common_indices].iloc[:, 0])[0, 1]
        else:
            corr = np.nan

        # Perform ADF test on the non-null differenced series
        adf_input = differenced_series.iloc[:, 0].dropna()
        if len(adf_input) > 1: # ADF test requires more than one observation
            # Unpack the ADF tuple properly
            adf_stat, p_val, _, _, critical_values, _ = adfuller(adf_input, regression='c', autolag='AIC')  # type: ignore
            conf_95=critical_values['5%']
            # Originally:
            # adf_result = adfuller(adf_input, regression='c', autolag='AIC')
            # p_val = adf_result[1]
            # adf_stat = adf_result[0]
            # conf_95 = adf_result[4]['5%']
        else:
            p_val = np.nan
            adf_stat = np.nan
            conf_95 = np.nan

        results.append({
            'd': d,
            'adfStat': adf_stat,
            'pVal': p_val,
            'corr': corr,
            'critical_val_95': conf_95
        })
        
    return pd.DataFrame(results)

def generate_fractional_series(ticker, min_d):
    """
    Generates the final fractionally differenced series using the optimal 'd' and saves it.
    
    Args:
        ticker (str): The ticker symbol to process.
        min_d (float): The optimal differencing order 'd'.
    """
    print(f"Generating final series for {ticker} with d={min_d:.2f}...")
    log_prices = dh.compute_log_prices(ticker=ticker)
    
    if not log_prices.empty:
        frac_diff_series = frac_diff_ffd(log_prices, d=min_d)
        output_path = f'fractional_series/{ticker}_fracdiff.csv'
        frac_diff_series.to_csv(output_path)
        print(f'Final fractionally differenced series saved to: {output_path}')
        return frac_diff_series