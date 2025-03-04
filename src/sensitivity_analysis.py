#!/usr/bin/env python3
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import logging

from settings import *
from auxiliary_functions import (
    calculate_opening_range,
    apply_orb_strategy,
    preprocess_data,
    check_available_tickers
)
from metrics import compute_performance_metrics


def sensitivity_analysis():
    """
    Perform sensitivity analysis for the ORB strategy on two dimensions:

    1. Volatility Regimes:
       Instead of relying on a VIX dataset, we compute a proxy daily volatility from each
       ticker's <close> price returns using a 20-day rolling standard deviation. We then
       classify each observation into three regimes based on quantile thresholds:
         - High volatility: rolling volatility >= 67th percentile
         - Medium volatility: between 33rd and 67th percentiles
         - Low volatility: rolling volatility <= 33rd percentile

    2. Liquidity Groups:
       We compute the average daily volume for each ticker and use the 90th and 10th percentiles
       to group tickers into high liquidity (top 10%) and low liquidity (bottom 10%).
    """
    # ------------------------
    # Volatility Regime Sensitivity Analysis (Using proxy volatility)
    # ------------------------
    performance_vol = {"High": [], "Medium": [], "Low": []}
    ticker_vols = []  # To store average daily volume for liquidity analysis
    ticker_data_dict = {}
    all_vols = []

    # Get list of available tickers
    available_tickers = check_available_tickers(NSE_AVAILABLE_EQUITY_LIST_PATH)

    # Load and preprocess data for each ticker; compute rolling volatility proxy
    for ticker in available_tickers:
        ticker_base = ticker.split(".")[0]
        try:
            data = preprocess_data(ticker_base, NSE_EQUITY_RAW_DATA_1MIN_DIR, YEARS, MONTHS)
            # Compute daily returns from the "<close>" column
            data["daily_return"] = data["<close>"].pct_change()
            # Compute 20-day rolling volatility as a proxy for daily volatility
            data["rolling_vol"] = data["daily_return"].rolling(window=20, min_periods=1).std()
            ticker_data_dict[ticker_base] = data
            # Append all computed rolling volatilities for threshold determination
            all_vols.extend(data["rolling_vol"].dropna().tolist())
        except Exception as e:
            logging.error(f"Failed processing data for ticker {ticker_base}: {e}")
            continue

    if not all_vols:
        print("No volatility data computed; aborting sensitivity analysis.")
        return

    # Define volatility thresholds based on the overall quantiles from the computed values.
    vol_high_threshold = np.percentile(all_vols, 67)
    vol_low_threshold = np.percentile(all_vols, 33)
    print(
        f"Proxy volatility thresholds (rolling vol): High >= {vol_high_threshold:.4f}, Low <= {vol_low_threshold:.4f}")

    def classify_vol(vol):
        if pd.isna(vol):
            return np.nan
        elif vol >= vol_high_threshold:
            return "High"
        elif vol <= vol_low_threshold:
            return "Low"
        else:
            return "Medium"

    # Process each ticker's data to extract performance metrics in each volatility regime.
    for ticker, data in ticker_data_dict.items():
        print(f"process {ticker} to extract performance metrics in each volatility regime.")
        # Map the computed rolling volatility to a regime label.
        data["vol_regime"] = data["rolling_vol"].apply(classify_vol)

        # For liquidity sensitivity, compute average daily volume using the "<volume>" column.
        try:
            avg_daily_volume = data.groupby("<date>")["<volume>"].sum().mean()
        except Exception as e:
            logging.error(f"Error calculating average volume for ticker {ticker}: {e}")
            avg_daily_volume = np.nan
        ticker_vols.append((ticker, avg_daily_volume))

        # For each volatility regime, run the ORB strategy with one default set of parameters.
        for regime in ["High", "Medium", "Low"]:
            subset = data[data["vol_regime"] == regime]
            if subset.empty:
                continue

            # Use default parameters from settings (first element of each list)
            pt_mult = PROFIT_TARGET_MULTIPLIERS[0]
            sl_mult = STOP_LOSS_MULTIPLIERS[0]
            or_minutes = OPENING_RANGE_MINUTES_LIST[0]
            m_open_time = MARKET_OPEN_TIMES[0]

            subset_or = calculate_opening_range(subset.copy(), or_minutes, m_open_time)
            df_orb = apply_orb_strategy(subset_or.copy(), pt_mult, sl_mult)
            subset_or["daily_returns"] = subset_or["<close>"].pct_change().fillna(0)
            subset_or["ORB_position"] = df_orb["signal"].shift().fillna(0)
            subset_or["ORB_returns"] = subset_or["ORB_position"] * subset_or["daily_returns"]
            subset_or["ORB_cum"] = (1 + subset_or["ORB_returns"]).cumprod()
            trading_days = subset_or.shape[0]
            if trading_days == 0:
                continue

            metrics = compute_performance_metrics(subset_or["ORB_cum"], subset_or["ORB_returns"], trading_days)
            performance_vol[regime].append(metrics)

    print("\n=== Volatility Regime Sensitivity Analysis ===")
    for regime, metrics_list in performance_vol.items():
        if metrics_list:
            avg_metrics = {
                k: np.mean([m[k] for m in metrics_list if not np.isnan(m[k])])
                for k in metrics_list[0].keys()
            }
            print(f"Regime: {regime}, Average Metrics: {avg_metrics}")
        else:
            print(f"Regime: {regime} has no data.")

    # ------------------------
    # Liquidity Sensitivity Analysis
    # ------------------------
    volumes_df = pd.DataFrame(ticker_vols, columns=["ticker", "avg_volume"])
    top_threshold = volumes_df["avg_volume"].quantile(0.9)
    bottom_threshold = volumes_df["avg_volume"].quantile(0.1)

    top_tickers = volumes_df[volumes_df["avg_volume"] >= top_threshold]["ticker"].tolist()
    bottom_tickers = volumes_df[volumes_df["avg_volume"] <= bottom_threshold]["ticker"].tolist()

    performance_liquidity = {"High_Liquidity": [], "Low_Liquidity": []}

    for ticker, data in ticker_data_dict.items():
        pt_mult = PROFIT_TARGET_MULTIPLIERS[0]
        sl_mult = STOP_LOSS_MULTIPLIERS[0]
        or_minutes = OPENING_RANGE_MINUTES_LIST[0]
        m_open_time = MARKET_OPEN_TIMES[0]

        data_or = calculate_opening_range(data.copy(), or_minutes, m_open_time)
        df_orb = apply_orb_strategy(data_or.copy(), pt_mult, sl_mult)
        data_or["daily_returns"] = data_or["<close>"].pct_change().fillna(0)
        data_or["ORB_position"] = df_orb["signal"].shift().fillna(0)
        data_or["ORB_returns"] = data_or["ORB_position"] * data_or["daily_returns"]
        data_or["ORB_cum"] = (1 + data_or["ORB_returns"]).cumprod()
        trading_days = data_or.shape[0]
        if trading_days == 0:
            continue

        metrics = compute_performance_metrics(data_or["ORB_cum"], data_or["ORB_returns"], trading_days)

        if ticker in top_tickers:
            performance_liquidity["High_Liquidity"].append(metrics)
        elif ticker in bottom_tickers:
            performance_liquidity["Low_Liquidity"].append(metrics)

    print("\n=== Liquidity Sensitivity Analysis ===")
    for group, metrics_list in performance_liquidity.items():
        if metrics_list:
            avg_metrics = {
                k: np.mean([m[k] for m in metrics_list if not np.isnan(m[k])])
                for k in metrics_list[0].keys()
            }
            print(f"Liquidity Group: {group}, Average Metrics: {avg_metrics}")
        else:
            print(f"Liquidity Group: {group} has no data.")


if __name__ == "__main__":
    sensitivity_analysis()