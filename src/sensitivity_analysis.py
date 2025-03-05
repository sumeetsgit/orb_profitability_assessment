#!/usr/bin/env python3
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import logging
from joblib import Parallel, delayed

# Optional: Use this snippet to profile the run
# import cProfile, pstats, io
# pr = cProfile.Profile()

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


def compute_orb_metrics(data, pt_mult, sl_mult, or_minutes, m_open_time):
    """
    Compute ORB strategy metrics on the data.
    """
    data_or = calculate_opening_range(data.copy(), or_minutes, m_open_time)
    df_orb = apply_orb_strategy(data_or.copy(), pt_mult, sl_mult)
    data_or["daily_returns"] = data_or["<close>"].pct_change().fillna(0)
    data_or["ORB_position"] = df_orb["signal"].shift().fillna(0)
    data_or["ORB_returns"] = data_or["ORB_position"] * data_or["daily_returns"]
    data_or["ORB_cum"] = (1 + data_or["ORB_returns"]).cumprod()
    trading_days = data_or.shape[0]
    if trading_days == 0:
        return None
    return compute_performance_metrics(data_or["ORB_cum"], data_or["ORB_returns"], trading_days)


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

    Results for both parts are stored to CSV for later analysis.
    """
    logging.info("Starting sensitivity analysis...")

    ###################################################################
    # STEP 1: Process all tickers and compute rolling volatility in parallel
    ###################################################################
    ticker_data_dict = {}
    all_vols = []

    available_tickers = check_available_tickers(NSE_AVAILABLE_EQUITY_LIST_PATH)
    logging.info(f"Found {len(available_tickers)} tickers available for processing.")

    def process_ticker_vol(ticker):
        ticker_base = ticker.split(".")[0]
        try:
            data = preprocess_data(ticker_base, NSE_EQUITY_RAW_DATA_1MIN_DIR, YEARS, MONTHS)
            data["daily_return"] = data["<close>"].pct_change()
            data["rolling_vol"] = data["daily_return"].rolling(window=20, min_periods=1).std()
            # Return tuple (ticker, data)
            return (ticker_base, data)
        except Exception as e:
            logging.error(f"Error processing ticker {ticker_base}: {e}")
            return None

    # Run in parallel across tickers using all available cores.
    results = Parallel(n_jobs=-1)(
        delayed(process_ticker_vol)(ticker) for ticker in
        tqdm(available_tickers, desc="Processing tickers for volatility analysis")
    )
    # Filter out errors
    for res in results:
        if res is not None:
            ticker_base, data = res
            ticker_data_dict[ticker_base] = data
            vols = data["rolling_vol"].dropna().tolist()
            all_vols.extend(vols)

    if not all_vols:
        logging.error("No volatility data computed; aborting sensitivity analysis.")
        print("No volatility data computed; aborting sensitivity analysis.")
        return

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

    ###################################################################
    # STEP 2: Analyze volatility regimes (process each ticker in parallel)
    ###################################################################
    def process_volatility_regime(ticker, data):
        # Add vol_regime column
        data["vol_regime"] = data["rolling_vol"].apply(classify_vol)
        results = []
        for regime in ["High", "Medium", "Low"]:
            subset = data[data["vol_regime"] == regime]
            if subset.empty:
                continue
            m = compute_orb_metrics(subset, PROFIT_TARGET_MULTIPLIERS[0], STOP_LOSS_MULTIPLIERS[0],
                                    OPENING_RANGE_MINUTES_LIST[0], MARKET_OPEN_TIMES[0])
            if m is not None:
                # Add ticker and regime information
                m["ticker"] = ticker
                m["vol_regime"] = regime
                results.append(m)
        return results

    vol_results_parallel = Parallel(n_jobs=-1)(
        delayed(process_volatility_regime)(ticker, data)
        for ticker, data in tqdm(ticker_data_dict.items(), desc="Analyzing volatility regimes")
    )
    # Flatten the list of lists
    vol_results = [item for sublist in vol_results_parallel for item in sublist]

    # Group metrics by volatility regime to compute average metrics.
    regimes = ["High", "Medium", "Low"]
    volatility_results = []
    for regime in regimes:
        metrics_list = [m for m in vol_results if m["vol_regime"] == regime]
        if metrics_list:
            avg_metrics = {k: safe_mean([m[k] for m in metrics_list])
                           for k in metrics_list[0].keys() if k not in ("ticker", "vol_regime")}
            avg_metrics["vol_regime"] = regime
            volatility_results.append(avg_metrics)
            print(f"Regime: {regime}, Average Metrics: {avg_metrics}")
            logging.info(f"Volatility Regime {regime} - Average Metrics: {avg_metrics}")
        else:
            print(f"Regime: {regime} has no data.")
            logging.info(f"Volatility Regime {regime} has no data.")

    # Save volatility sensitivity results
    vol_df = pd.DataFrame(volatility_results)
    vol_csv_path = Path("..") / "data" / "output" / "volatility_sensitivity.csv"
    vol_df.to_csv(vol_csv_path, index=False)
    logging.info(f"Volatility sensitivity analysis results saved to {vol_csv_path}")
    print(f"Volatility sensitivity analysis results saved to {vol_csv_path}")

    ###################################################################
    # STEP 3: LIQUIDITY SENSITIVITY ANALYSIS
    ###################################################################
    # Build a DataFrame of average daily volumes for each ticker.
    ticker_vols = []
    for ticker, data in ticker_data_dict.items():
        try:
            avg_daily_volume = data.groupby("<date>")["<volume>"].sum().mean()
        except Exception as e:
            logging.error(f"Error computing average volume for ticker {ticker}: {e}")
            avg_daily_volume = np.nan
        ticker_vols.append((ticker, avg_daily_volume))
    volumes_df = pd.DataFrame(ticker_vols, columns=["ticker", "avg_volume"])
    top_threshold = volumes_df["avg_volume"].quantile(0.9)
    bottom_threshold = volumes_df["avg_volume"].quantile(0.1)
    top_tickers = volumes_df[volumes_df["avg_volume"] >= top_threshold]["ticker"].tolist()
    bottom_tickers = volumes_df[volumes_df["avg_volume"] <= bottom_threshold]["ticker"].tolist()
    logging.info(f"Liquidity thresholds determined: Top 10% >= {top_threshold}, Bottom 10% <= {bottom_threshold}")

    def process_liquidity(ticker, data):
        # Compute ORB metrics on the entire ticker data (no regime split)
        m = compute_orb_metrics(data, PROFIT_TARGET_MULTIPLIERS[0],
                                STOP_LOSS_MULTIPLIERS[0],
                                OPENING_RANGE_MINUTES_LIST[0],
                                MARKET_OPEN_TIMES[0])
        return m

    liquidity_metrics = Parallel(n_jobs=-1)(
        delayed(process_liquidity)(ticker, data)
        for ticker, data in tqdm(ticker_data_dict.items(), desc="Analyzing liquidity")
    )
    # Build performance_liquidity dict based on ticker liquidity groups.
    performance_liquidity = {"High_Liquidity": [], "Low_Liquidity": []}
    for (ticker, _), orb_metric in zip(ticker_vols, liquidity_metrics):
        # If ORB metric is None, skip.
        if orb_metric is None:
            continue
        if ticker in top_tickers:
            performance_liquidity["High_Liquidity"].append(orb_metric)
            logging.info(f"Ticker {ticker} assigned to High Liquidity group.")
        elif ticker in bottom_tickers:
            performance_liquidity["Low_Liquidity"].append(orb_metric)
            logging.info(f"Ticker {ticker} assigned to Low Liquidity group.")
        else:
            logging.info(f"Ticker {ticker} is not in Top or Bottom liquidity group.")

    print("\n=== Liquidity Sensitivity Analysis ===")
    liquidity_results = []
    for group, metrics_list in performance_liquidity.items():
        if metrics_list:
            avg_metrics = {k: safe_mean([m[k] for m in metrics_list])
                           for k in metrics_list[0].keys() if k != "ticker"}
            avg_metrics["liquidity_group"] = group
            liquidity_results.append(avg_metrics)
            print(f"Liquidity Group: {group}, Average Metrics: {avg_metrics}")
            logging.info(f"Liquidity Group {group} - Average Metrics: {avg_metrics}")
        else:
            print(f"Liquidity Group: {group} has no data.")
            logging.info(f"Liquidity Group {group} has no data.")

    liq_df = pd.DataFrame(liquidity_results)
    liq_csv_path = Path("..") / "data" / "output" / "liquidity_sensitivity.csv"
    liq_df.to_csv(liq_csv_path, index=False)
    logging.info(f"Liquidity sensitivity analysis results saved to {liq_csv_path}")
    print(f"Liquidity sensitivity analysis results saved to {liq_csv_path}")

    logging.info("Sensitivity analysis completed.")


if __name__ == "__main__":
    # Optional: If you want to run under cProfile, uncomment the following lines:
    # pr.enable()
    sensitivity_analysis()
    # pr.disable()
    # s = io.StringIO()
    # sortby = 'cumulative'
    # ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    # ps.print_stats()
    # print(s.getvalue())