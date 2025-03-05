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

def safe_mean(values):
    """
    Returns the arithmetic mean of the numeric values in the list.
    If a value cannot be converted to float, it is skipped.
    """
    numeric_vals = []
    for v in values:
        try:
            numeric_vals.append(float(v))
        except Exception:
            continue
    return np.mean(numeric_vals) if numeric_vals else np.nan

def sensitivity_analysis():
    """
    Perform a sensitivity analysis for the ORB strategy without an external VIX dataset.

    1. Volatility regime sensitivity:
       - Use a 20-day rolling standard deviation of daily returns from the <close> price as a
         proxy for volatility.
       - Define volatility regimes based on the overall quantiles:
             High: rolling volatility >= 67th percentile,
             Low: rolling volatility <= 33rd percentile,
             Medium: in between.

    2. Liquidity sensitivity:
       - Compute each ticker’s average daily volume using the <volume>.
       - Compare performance metrics between tickers in the top 10% (high liquidity) and bottom
         10% (low liquidity).

    Progress is monitored with tqdm progress bars and detailed logging.

    At the end, results for both sensitivity analyses are stored to CSV files.
    """
    logging.info("Starting sensitivity analysis...")

    # ------------------------
    # VOLATILITY REGIME SENSITIVITY ANALYSIS (Using rolling volatility as proxy)
    # ------------------------
    performance_vol = {"High": [], "Medium": [], "Low": []}
    ticker_vols = []  # To store average daily volume (for liquidity grouping)
    ticker_data_dict = {}
    all_vols = []

    # Get list of available tickers.
    available_tickers = check_available_tickers(NSE_AVAILABLE_EQUITY_LIST_PATH)
    logging.info(f"Found {len(available_tickers)} tickers available for processing.")

    # Process each ticker and compute rolling volatility.
    for ticker in tqdm(available_tickers, desc="Processing tickers for volatility analysis"):
        ticker_base = ticker.split(".")[0]
        try:
            data = preprocess_data(ticker_base, NSE_EQUITY_RAW_DATA_1MIN_DIR, YEARS, MONTHS)
            # Compute daily returns and a 20-day rolling volatility as a proxy.
            data["daily_return"] = data["<close>"].pct_change()
            data["rolling_vol"] = data["daily_return"].rolling(window=20, min_periods=1).std()
            ticker_data_dict[ticker_base] = data
            all_vols.extend(data["rolling_vol"].dropna().tolist())
            logging.info(f"Ticker {ticker_base}: Loaded and computed volatility for {data.shape[0]} records.")
        except Exception as e:
            logging.error(f"Error processing ticker {ticker_base}: {e}")
            continue

    if not all_vols:
        logging.error("No volatility data computed; aborting sensitivity analysis.")
        print("No volatility data computed; aborting sensitivity analysis.")
        return

    # Define volatility thresholds using overall quantiles.
    vol_high_threshold = np.percentile(all_vols, 67)
    vol_low_threshold = np.percentile(all_vols, 33)
    print(f"Proxy Volatility thresholds: High >= {vol_high_threshold:.4f}, Low <= {vol_low_threshold:.4f}")
    logging.info(f"Volatility thresholds: High >= {vol_high_threshold:.4f}, Low <= {vol_low_threshold:.4f}")

    def classify_vol(vol):
        if pd.isna(vol):
            return np.nan
        elif vol >= vol_high_threshold:
            return "High"
        elif vol <= vol_low_threshold:
            return "Low"
        else:
            return "Medium"

    # Analyze volatility regimes for each ticker.
    for ticker, data in tqdm(ticker_data_dict.items(), desc="Analyzing volatility regimes"):
        data["vol_regime"] = data["rolling_vol"].apply(classify_vol)

        # For liquidity sensitivity: compute the average daily volume using the "<volume>" field.
        try:
            avg_daily_volume = data.groupby("<date>")["<volume>"].sum().mean()
        except Exception as e:
            logging.error(f"Error computing average volume for ticker {ticker}: {e}")
            avg_daily_volume = np.nan
        ticker_vols.append((ticker, avg_daily_volume))

        # Run the ORB strategy for each volatility regime (if sufficient data exists).
        for regime in ["High", "Medium", "Low"]:
            subset = data[data["vol_regime"] == regime]
            if subset.empty:
                continue

            # Use default parameters: first elements from the multipliers and lists.
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
            logging.info(f"Ticker {ticker} regime {regime}: Calculated metrics.")

    print("\n=== Volatility Regime Sensitivity Analysis ===")
    volatility_results = []
    for regime, metrics_list in performance_vol.items():
        if metrics_list:
            avg_metrics = {k: safe_mean([m[k] for m in metrics_list]) for k in metrics_list[0].keys()}
            # Save average metrics per regime to the combined output list.
            avg_metrics["vol_regime"] = regime
            volatility_results.append(avg_metrics)
            print(f"Regime: {regime}, Average Metrics: {avg_metrics}")
            logging.info(f"Volatility Regime {regime} - Average Metrics: {avg_metrics}")
        else:
            print(f"Regime: {regime} has no data.")
            logging.info(f"Volatility Regime {regime} has no data.")

    # Save the volatility sensitivity analysis metrics to a CSV file.
    vol_df = pd.DataFrame(volatility_results)
    vol_csv_path = Path(".") / "data" / "output" / "volatility_sensitivity.csv"
    vol_df.to_csv(vol_csv_path, index=False)
    logging.info(f"Volatility sensitivity analysis results saved to {vol_csv_path}")
    print(f"Volatility sensitivity analysis results saved to {vol_csv_path}")

    # ------------------------
    # LIQUIDITY SENSITIVITY ANALYSIS
    # ------------------------
    volumes_df = pd.DataFrame(ticker_vols, columns=["ticker", "avg_volume"])
    top_threshold = volumes_df["avg_volume"].quantile(0.9)
    bottom_threshold = volumes_df["avg_volume"].quantile(0.1)
    top_tickers = volumes_df[volumes_df["avg_volume"] >= top_threshold]["ticker"].tolist()
    bottom_tickers = volumes_df[volumes_df["avg_volume"] <= bottom_threshold]["ticker"].tolist()
    logging.info(f"Liquidity thresholds determined: Top 10% >= {top_threshold}, Bottom 10% <= {bottom_threshold}")

    performance_liquidity = {"High_Liquidity": [], "Low_Liquidity": []}

    for ticker, data in tqdm(ticker_data_dict.items(), desc="Analyzing liquidity"):
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
            logging.info(f"Ticker {ticker} assigned to High Liquidity group.")
        elif ticker in bottom_tickers:
            performance_liquidity["Low_Liquidity"].append(metrics)
            logging.info(f"Ticker {ticker} assigned to Low Liquidity group.")
        else:
            logging.info(f"Ticker {ticker} not in Top or Bottom liquidity group.")

    print("\n=== Liquidity Sensitivity Analysis ===")
    liquidity_results = []
    for group, metrics_list in performance_liquidity.items():
        if metrics_list:
            avg_metrics = {k: safe_mean([m[k] for m in metrics_list]) for k in metrics_list[0].keys()}
            avg_metrics["liquidity_group"] = group
            liquidity_results.append(avg_metrics)
            print(f"Liquidity Group: {group}, Average Metrics: {avg_metrics}")
            logging.info(f"Liquidity Group {group} - Average Metrics: {avg_metrics}")
        else:
            print(f"Liquidity Group: {group} has no data.")
            logging.info(f"Liquidity Group {group} has no data.")

    # Save the liquidity sensitivity analysis metrics to a CSV file.
    liq_df = pd.DataFrame(liquidity_results)
    liq_csv_path = Path(".") / "data" / "output" / "liquidity_sensitivity.csv"
    liq_df.to_csv(liq_csv_path, index=False)
    logging.info(f"Liquidity sensitivity analysis results saved to {liq_csv_path}")
    print(f"Liquidity sensitivity analysis results saved to {liq_csv_path}")

    logging.info("Sensitivity analysis completed.")

if __name__ == "__main__":
    sensitivity_analysis()