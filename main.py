# file: main.py
# !/usr/bin/env python3

# -*- coding: utf-8 -*-
'''
Copyright (c) 2025 Shaan Ali Remani. All rights reserved.
Licensed under the MIT License.
'''

import matplotlib.pyplot as plt
import fractional_differencing as fd
import data_handler as dh

# --- Config ---
ticker = '^GSPC' # Ticker of interest
adf_confLevel = 0.05 # p-value threshold for stationarity

# --- Run Analysis ---
print(f"Running fractional differencing analysis for {ticker}...")
# Call the fractional differencing analysis function
results_df = fd.find_minimum_d(ticker=ticker)

# --- Output ---
if not results_df.empty:
    print("\n--- Fractional Differencing Analysis Results ---")
    print(results_df)

    # Find and print the minimum d that passes the stationarity test
    try:
        min_d_row = results_df[results_df['pVal'] < adf_confLevel].sort_values('d').iloc[0]
        min_d = min_d_row['d']
        print(f"Minimum 'd' to achieve stationarity (p-value < {adf_confLevel}): {min_d:.2f}")

        # Generate and save final fractionally differenced series
        fractional_series=fd.generate_fractional_series(ticker=ticker, min_d=min_d)
    
    except IndexError:
        print(f"No differencing order achieved stationarity at the p < {adf_confLevel} level.")

# Plot results
fig, ax1 = plt.subplots(figsize=(12, 7))

# Plot correlation on primary y-axis
ax1.plot(results_df['d'], results_df['corr'], 'b-', marker='o', label='Correlation with Original Series')
ax1.set_xlabel('Differencing Order (d)')
ax1.set_ylabel('Correlation', color='b')
ax1.tick_params(axis='y', labelcolor='b')
ax1.grid(True, which='both', linestyle='--', linewidth=0.5)

# Create secondary y-axis for ADF statistic
ax2 = ax1.twinx()
ax2.plot(results_df['d'], results_df['adfStat'], 'r-', marker='x', label='ADF Statistic')
ax2.set_ylabel('ADF Statistic', color='r')
ax2.tick_params(axis='y', labelcolor='r')

# Plot 95% confidence level line for ADF test
ax2.axhline(y=results_df['critical_val_95'].mean(), color='grey', linestyle='--', label='95% Confidence Level')

# Final plot adjustments
fig.tight_layout(rect=(0, 0, 0.9, 1)) # Make room for the man, the myth, the 'Legend'
plt.title(f'Fractional Differencing Analysis for {ticker}')

# Combine legends from both axes
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc='upper right')

plt.show()

# # Quick Sanity Check: Plot the fully differenced series (i.e. returns) against fractionally differenced series

# # Compute log return
# logPrice=dh.compute_log_prices(ticker)
# logReturn = logPrice.diff().dropna()

# # Plotting the log returns and fractionally differenced series
# plt.figure(figsize=(12, 6))
# plt.plot(logReturn.index, logReturn, label='Log Returns', color='blue')
# plt.plot(fractional_series.index, fractional_series, label='Fractionally Differenced Series', color='orange') # type: ignore
# plt.title(f'Log Returns vs Fractionally Differenced Series for {ticker}')
# plt.xlabel('Date')
# plt.ylabel('Price')
# plt.legend()
# plt.show()


