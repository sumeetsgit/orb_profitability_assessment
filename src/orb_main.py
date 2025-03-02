import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import glob
from pathlib import Path
import os
import logging
from tqdm import tqdm
from datetime import datetime

from settings import *
from auxiliary_functions import *



# --- --------- Main Execution Flow with Grid Search for finding best params --------- ---
def main_gs():
    try:
        # Step 1: Setup Logging
        setup_logging(LOGS_DIR + RUN)
        logging.info("ORB Strategy Backtest Script Started.")


        # Step 2: Create Output Directory
        create_output_directory(OUTPUT_DIR + RUN)
        logging.info("Created output directory at - {}.".format(OUTPUT_DIR + RUN))

        # Step 3: Check available valid data
        available_nse_tickers = check_available_tickers(NSE_AVAILABLE_EQUITY_LIST_PATH)
        logging.info("Data is available for {} tickers.".format(len(available_nse_tickers)))



        complete_tickers = []
        incomplete_tickers = []


        # Step 4: Load and Process Data
        for ticker in tqdm(available_nse_tickers[:5]):
            logging.info("=====================================")
            logging.info("Processing data for ticker - {}.".format(ticker))
            # data = load_and_concatenate_csv(DATA_DIR)
            # data = prepare_data(data)

            ticker = ticker.split(".")[0]
            data = preprocess_data(ticker, NSE_EQUITY_RAW_DATA_1MIN_DIR, YEARS, MONTHS)

            rangewise_data = apply_orb_strategy_with_param_range(data, PROFIT_TARGET_MULTIPLIERS, STOP_LOSS_MULTIPLIERS, OPENING_RANGE_MINUTES_LIST, MARKET_OPEN_TIMES)
            # print(rangewise_data.keys())

            backtesting_results = backtest_strategies(rangewise_data)
            

            performance_metrics_df, best_parameters = get_optimized_parameters_gs(backtesting_results, CRITERIA_SHARPE_THRESHOLD, CRITERIA_MAX_DRAWDOWN)

            print(performance_metrics_df.shape)
            print("BEST PARAMETERS --> ", best_parameters)

            if best_parameters is not None:
                best_candidate_df = rangewise_data[best_parameters]
                best_bt_df = backtesting_results[best_parameters]
                strategies = {
                    "ORB": ("ORB_cum", "ORB_returns"),
                    "Buy and Hold": ("BH_cum", "BH_returns"),
                    "MA Crossover": ("MA_cum", "MA_returns"),
                    "Mean Reversion": ("MR_cum", "MR_returns"),
                    "Volatility Breakout": ("VB_cum", "VB_returns"),
                    "Intraday Momentum": ("MOM_cum", "MOM_returns"),
                    "Golden Ratio Breakout": ("GR_cum", "GR_returns")
                }
                metrics_dict = {}
                trading_days = best_bt_df.shape[0]

                for strat, (cum_col, ret_col) in strategies.items():
                    cum_series = best_bt_df[cum_col]
                    daily_ret = best_bt_df[ret_col]
                    metrics = compute_performance_metrics(cum_series, daily_ret, trading_days)
                    metrics_dict[strat] = metrics
                    # print(f"Strategy: {strat}")
                    for m_key, m_val in metrics.items():
                        print(f"  {m_key}: {m_val:.4f}")
                    print("-" * 40)
                
                plot_performance_comparison(metrics_dict)
            else:
                print("No parameter combination met the optimization criteria.")



            
            
            # import sys
            # sys.exit()



            # Step 7: Backtest Strategies
            # cumulative_returns_orb, cumulative_returns_bh = backtest_strategies(data)
            # logging.info("Completed backtesting of both ORB and Buy and Hold strategies.")

            # sample_key = list(rangewise_data.keys())[0]
            # sample_data = rangewise_data[sample_key]
            # results = backtest_strategies(sample_data)



            # # Step 8: Calculate Performance Metrics
            # daily_returns_orb = data['strategy_returns_orb'].dropna()
            # daily_returns_bh = data['strategy_returns_bh'].dropna()

            # metrics_orb = calculate_performance_metrics(cumulative_returns_orb.dropna(), daily_returns_orb)
            # metrics_bh = calculate_performance_metrics(cumulative_returns_bh.dropna(), daily_returns_bh)

            # # Compile performance metrics into DataFrame
            # performance_df = pd.DataFrame({
            #     'ORB Strategy': metrics_orb,
            #     'Buy and Hold': metrics_bh
            # })

            # logging.info("Calculated performance metrics for both strategies.")
            # print("\nPerformance Metrics Comparison:")
            # print(performance_df)

            # # Step 8: Plotting
            # plot_cumulative_returns(cumulative_returns_orb, cumulative_returns_bh, OUTPUT_DIR, ticker)
            # plot_drawdowns(cumulative_returns_orb, cumulative_returns_bh, OUTPUT_DIR, ticker)
            # plot_performance_metrics(performance_df, OUTPUT_DIR, ticker)

            # # Step 9: Save Performance Metrics to CSV
            # save_performance_metrics(performance_df, OUTPUT_DIR, ticker)

            # # Step 10: Save the Enhanced Dataset
            # save_enhanced_dataset(data, OUTPUT_DIR, ticker)

            # logging.info("ORB Strategy Backtest Script completed successfully.")
            # complete_tickers.append(ticker)

    except Exception as e:
        logging.error(f"An error occurred during execution: {e}")
        print(f"An error occurred: {e}")
        incomplete_tickers.append(ticker)
        complete_tickers_df = pd.DataFrame(complete_tickers)
        complete_tickers_df.to_csv("complete_tickers.csv")




# --- --------- Main Execution Flow with Genetic Algorithm for finding best params --------- ---
def main_ga():
    try:
        # Step 1: Setup Logging
        setup_logging(LOGS_DIR + RUN)
        logging.info("ORB Strategy Backtest Script Started.")


        # Step 2: Create Output Directory
        create_output_directory(OUTPUT_DIR + RUN)
        logging.info("Created output directory at - {}.".format(OUTPUT_DIR + RUN))

        # Step 3: Check available valid data
        available_nse_tickers = check_available_tickers(NSE_AVAILABLE_EQUITY_LIST_PATH)
        logging.info("Data is available for {} tickers.".format(len(available_nse_tickers)))



        complete_tickers = []
        incomplete_tickers = []

        # Step 4: Load and Prepare Data
        for ticker in tqdm(available_nse_tickers):
            logging.info("=====================================")
            logging.info("Processing data for ticker - {}.".format(ticker))
            # data = load_and_concatenate_csv(DATA_DIR)
            # data = prepare_data(data)

            ticker = ticker.split(".")[0]
            data = preprocess_data(ticker, NSE_EQUITY_RAW_DATA_1MIN_DIR, YEARS, MONTHS)


            best_candidate, best_fitness = evolve_parameters(data)

            print("BEST CANDIDATE --> ", best_candidate)
            print("BEST FITNESS --> ", best_fitness)

            
            import sys
            sys.exit()

            # Step 8: Calculate Performance Metrics
            daily_returns_orb = data['strategy_returns_orb'].dropna()
            daily_returns_bh = data['strategy_returns_bh'].dropna()

            metrics_orb = calculate_performance_metrics(cumulative_returns_orb.dropna(), daily_returns_orb)
            metrics_bh = calculate_performance_metrics(cumulative_returns_bh.dropna(), daily_returns_bh)

            # Compile performance metrics into DataFrame
            performance_df = pd.DataFrame({'ORB Strategy': metrics_orb, 'Buy and Hold': metrics_bh})

            logging.info("Calculated performance metrics for both strategies.")
            print("\nPerformance Metrics Comparison:")
            print(performance_df)

            # Step 8: Plotting
            plot_cumulative_returns(cumulative_returns_orb, cumulative_returns_bh, OUTPUT_DIR, ticker)
            plot_drawdowns(cumulative_returns_orb, cumulative_returns_bh, OUTPUT_DIR, ticker)
            plot_performance_metrics(performance_df, OUTPUT_DIR, ticker)

            # Step 9: Save Performance Metrics to CSV
            save_performance_metrics(performance_df, OUTPUT_DIR, ticker)

            # Step 10: Save the Enhanced Dataset
            save_enhanced_dataset(data, OUTPUT_DIR, ticker)

            logging.info("ORB Strategy Backtest Script completed successfully.")
            complete_tickers.append(ticker)

    except Exception as e:
        logging.error(f"An error occurred during execution: {e}")
        print(f"An error occurred: {e}")
        incomplete_tickers.append(ticker)
        complete_tickers_df = pd.DataFrame(complete_tickers)
        complete_tickers_df.to_csv("complete_tickers.csv")


# Execute the main function
if __name__ == "__main__":
    main_gs()