import os
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random


from settings import *




# --- --------- Create Output Directory --------- ---
def create_output_directory(output_dir):
    """
    Create the output directory if it doesn't exist.

    Parameters:
    - output_dir (Path): Path to the output directory.
    """
    try:
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        print(f"Created output directory at - {path.resolve()}")
    except Exception as e:
        print(f"Failed to create output directory at {output_dir}: {e}")
        raise e
    


# --- --------- Setup Logging --------- ---
def setup_logging(logs_dir):
    """
    Configure logging to capture the script's execution details.

    Parameters:
    - logs_dir (Path): Directory where the log file will be saved.
    """
    log_file = logs_dir + '_orb_backtest.log'
    logging.basicConfig(
        filename=log_file,
        filemode='a',
        format='%(asctime)s %(levelname)s:%(message)s',
        level=logging.INFO
    )
    logging.info("Logging has been configured.")



def check_available_tickers(required_path):
    available_nse_tickers_df = pd.read_csv(required_path)
    available_nse_tickers = list(available_nse_tickers_df["tickers"])
    print("Data is available for {} tickers.".format(len(available_nse_tickers)))
    return available_nse_tickers


# --- --------- Load and Concatenate CSV Files --------- ---
def load_and_concatenate_csv(data_dir, file_pattern='NIFTY_*.csv'):
    """
    Load multiple CSV files matching the given pattern and concatenate them into a single DataFrame.

    Parameters:
    - data_dir (Path): Directory containing the CSV files.
    - file_pattern (str): Glob pattern to match CSV files.

    Returns:
    - pd.DataFrame: Concatenated DataFrame containing all data.
    """
    csv_files = sorted(data_dir.glob(file_pattern))

    if not csv_files:
        logging.error(f"No CSV files found in {data_dir} matching the pattern '{file_pattern}'.")
        raise FileNotFoundError(f"No CSV files found in {data_dir} matching the pattern '{file_pattern}'.")

    df_list = []
    for file in csv_files:
        try:
            logging.info(f"Loading {file.name}...")
            df = pd.read_csv(file, parse_dates=['datetime'])

            # Ensure required columns are present
            required_columns = {'datetime', 'open', 'high', 'low', 'close', 'volume'}
            if not required_columns.issubset(df.columns):
                missing = required_columns - set(df.columns)
                logging.error(f"CSV file {file.name} is missing columns: {missing}")
                raise ValueError(f"CSV file {file.name} is missing columns: {missing}")

            df.set_index('datetime', inplace=True)
            df_list.append(df)
        except Exception as e:
            logging.error(f"Error loading {file.name}: {e}")
            raise e

    # Concatenate all DataFrames
    data = pd.concat(df_list)
    data.sort_index(inplace=True)
    logging.info(f"Total records after concatenation: {len(data)}")

    return data




# --- --------- Prepare Data --------- ---
def prepare_data(data):
    """
    Prepare the DataFrame by adding necessary columns for grouping and strategy calculations.

    Parameters:
    - data (pd.DataFrame): The concatenated DataFrame.

    Returns:
    - pd.DataFrame: Prepared DataFrame.
    """
    # Add 'date' column for grouping
    data['date'] = data.index.date
    logging.info("Added 'date' column for grouping.")

    return data


def preprocess_data(ticker, raw_data_dir, years, months):
    print("TICKER --> ", ticker)
    ticker_dfs = []
    for year in years:
        for month in months:
            absolute_ticker_data_path = str(raw_data_dir) + "/" + year + "/" + month + "_" + year + "/" + ticker + ".csv"
            df = pd.read_csv(absolute_ticker_data_path)
            ticker_dfs.append(df)

    data = pd.concat(ticker_dfs)

    data["<datetime>"] = data["<date>"].astype(str) + " " + data["<time>"].astype(str)
    data['<datetime>'] = pd.to_datetime(data['<datetime>'])
    data = data[["<ticker>", "<date>", "<time>", "<datetime>", "<open>", "<high>", "<low>", "<close>", "<volume>"]]
    data.set_index('<datetime>', inplace=True)
    data['<date>'] = data.index.date
    data.sort_index(inplace=True)
    print(data.shape)
    # data.to_csv("temp.csv")
    logging.info(f"Total records after concatenation for {ticker}: {len(data)}")

    return data

    


# --- --------- Calculate Opening Range --------- ---
def calculate_opening_range(data, opening_range_minutes, market_open_time):
    """
    Calculate the opening range high and low for each trading day.

    Parameters:
    - data (pd.DataFrame): Prepared DataFrame.
    - opening_range_minutes (int): Duration of the opening range in minutes.
    - market_open_time (str): Market opening time in HH:MM:SS format.

    Returns:
    - pd.DataFrame: DataFrame with opening range high and low.
    """
    # Define threshold time
    base_time = pd.Timestamp(market_open_time)
    threshold_timestamp = base_time + pd.Timedelta(minutes=opening_range_minutes)
    threshold_time = threshold_timestamp.time()
    logging.info(f"Opening Range Threshold Time: {threshold_time}")

    # Filter data within the opening range using groupby and apply
    try:
        opening_range = data.groupby('<date>').apply(
            lambda x: x[x.index.time <= threshold_time]
        ).reset_index(level=0, drop=True)
        logging.info(f"Total records within the opening range: {len(opening_range)}")
    except FutureWarning as fw:
        logging.warning(f"FutureWarning encountered: {fw}")
        # Handle the warning by modifying the groupby apply as needed
        opening_range = data.groupby('<date>').filter(
            lambda x: (x.index.time <= threshold_time).any()
        ).copy()
        opening_range = opening_range.groupby('<date>').apply(
            lambda x: x[x.index.time <= threshold_time]
        ).reset_index(drop=True)
        logging.info(f"Total records within the opening range after handling warning: {len(opening_range)}")

    # Calculate opening range high and low
    opening_range_high = opening_range.groupby('<date>')['<high>'].max()
    opening_range_low = opening_range.groupby('<date>')['<low>'].min()

    # Merge back into main data
    data['opening_range_high'] = data['<date>'].map(opening_range_high)
    data['opening_range_low'] = data['<date>'].map(opening_range_low)
    logging.info("Merged opening range high and low back into the main DataFrame.")

    return data


# --- --------- Define ORB Strategy --------- ---
def apply_orb_strategy(df, profit_target_multiplier, stop_loss_multiplier):
    """
    Apply the Opening Range Breakout strategy to the DataFrame.

    Parameters:
    - df (pd.DataFrame): DataFrame with opening range high and low.
    - profit_target_multiplier (float): Multiplier for profit targets.
    - stop_loss_multiplier (float): Multiplier for stop-loss levels.

    Returns:
    - pd.DataFrame: DataFrame with strategy signals and levels.
    """
    # Generate entry signals
    # print(df.columns)
    df['long_entry'] = df['<close>'] > df['opening_range_high']
    df['short_entry'] = df['<close>'] < df['opening_range_low']
    logging.info("Generated long and short entry signals.")

    # Calculate the range size
    df['range_size'] = df['opening_range_high'] - df['opening_range_low']
    logging.info("Calculated range size for each date.")

    # Define profit targets and stop losses for long positions
    df['profit_target_long'] = df['opening_range_high'] + profit_target_multiplier * df['range_size']
    df['stop_loss_long'] = df['opening_range_high'] - stop_loss_multiplier * df['range_size']

    # Define profit targets and stop losses for short positions
    df['profit_target_short'] = df['opening_range_low'] - profit_target_multiplier * df['range_size']
    df['stop_loss_short'] = df['opening_range_low'] + stop_loss_multiplier * df['range_size']
    logging.info("Defined profit targets and stop-loss levels for both long and short positions.")

    # Initialize signals
    df['signal'] = 0

    # Assign signals based on entry conditions
    df.loc[df['long_entry'], 'signal'] = 1  # Buy signal
    df.loc[df['short_entry'], 'signal'] = -1  # Sell signal
    logging.info("Assigned signals based on entry conditions.")

    # Evaluate trade outcome on the same bar (for demonstration):
    # For long trades: win if the bar's high reaches the profit target; loss if the bar's low hits the stop loss.
    long_win_condition = (df['long_entry']) & (df['<high>'] >= df['profit_target_long'])
    long_loss_condition = (df['long_entry']) & (df['<low>'] <= df['stop_loss_long'])

    # For short trades: win if the bar's low reaches the profit target; loss if the bar's high hits the stop loss.
    short_win_condition = (df['short_entry']) & (df['<low>'] <= df['profit_target_short'])
    short_loss_condition = (df['short_entry']) & (df['<high>'] >= df['stop_loss_short'])

    df['trade_outcome'] = np.nan
    df['trade_outcome'] = df['trade_outcome'].astype(object)
    df.loc[long_win_condition, 'trade_outcome'] = 'win'
    df.loc[long_loss_condition, 'trade_outcome'] = 'loss'
    df.loc[short_win_condition, 'trade_outcome'] = 'win'
    df.loc[short_loss_condition, 'trade_outcome'] = 'loss'

    print("ORB strategy applied with parameters - Profit Target Multiplier: {}, Stop Loss Multiplier: {}".format(profit_target_multiplier, stop_loss_multiplier))

    return df


def apply_orb_strategy_with_param_range(data, profit_target_multipliers, stop_loss_multipliers, opening_range_minutes_list, market_open_times):
    """
    Run the ORB strategy over a range of parameters.
    
    For each combination of:
       - market_open_time (e.g., "09:15:00", "09:30:00")
       - opening_range_minutes (e.g., 15, 30)
       - profit_target_multiplier (e.g., 1.5, 2.0, 2.5)
       - stop_loss_multiplier (e.g., 0.5, 1.0, 1.5)
       
    This function:
      a) Computes the opening range,
      b) Applies the ORB strategy with trade evaluation,
      c) Returns a dictionary where keys describe the parameter combination and values are the resulting DataFrame.
    
    Parameters:
      - data (pd.DataFrame): The raw data with required columns.
      - profit_target_multipliers (list of float)
      - stop_loss_multipliers (list of float)
      - opening_range_minutes_list (list of int)
      - market_open_times (list of str)
      
    Returns:
      - dict: Keys are parameter combination strings, values are DataFrames with ORB strategy results.
    """
    data_dict = {}
    for market_open_time in market_open_times:
        for opening_range in opening_range_minutes_list:
            # Compute the opening range for this configuration
            data_or = calculate_opening_range(data.copy(), opening_range, market_open_time)
            for pt_mult in profit_target_multipliers:
                for sl_mult in stop_loss_multipliers:
                    df_orb = apply_orb_strategy(data_or.copy(), profit_target_multiplier=pt_mult, stop_loss_multiplier=sl_mult)
                    key = f"mo:{market_open_time}_or:{opening_range}_pt:{pt_mult}_sl:{sl_mult}"
                    data_dict[key] = df_orb
                    logging.info(f"Processed {key}")

    return data_dict


# # --- --------- Backtest Strategies --------- ---
# def backtest_strategies(df):
#     """
#     Backtest ORB and Buy and Hold strategies.

#     Parameters:
#     - df (pd.DataFrame): DataFrame with strategy signals.

#     Returns:
#     - pd.Series, pd.Series: Cumulative returns for ORB and Buy and Hold strategies.
#     """
#     # --- Opening Range Breakout Strategy ---

#     # Shift signals to represent positions taken at the next time step
#     df['position_orb'] = df['signal'].shift()
#     logging.info("Shifted ORB signals to represent next time step positions.")

#     # Calculate daily returns based on closing prices
#     df['daily_returns'] = df['<close>'].pct_change()

#     # Strategy returns: position * daily returns
#     df['strategy_returns_orb'] = df['position_orb'] * df['daily_returns']
#     logging.info("Calculated strategy returns for ORB.")

#     # Calculate cumulative returns for ORB
#     cumulative_returns_orb = (1 + df['strategy_returns_orb']).cumprod() - 1
#     logging.info("Calculated cumulative returns for ORB.")

#     # --- Buy and Hold Strategy ---

#     # Define Buy and Hold position (always 1)
#     df['position_bh'] = 1
#     logging.info("Defined Buy and Hold position.")

#     # Strategy returns for Buy and Hold
#     df['strategy_returns_bh'] = df['position_bh'] * df['daily_returns']
#     logging.info("Calculated strategy returns for Buy and Hold.")

#     # Calculate cumulative returns for Buy and Hold
#     cumulative_returns_bh = (1 + df['strategy_returns_bh']).cumprod() - 1
#     logging.info("Calculated cumulative returns for Buy and Hold.")

#     return cumulative_returns_orb, cumulative_returns_bh


def backtest_strategies(df):
    """
    Backtest multiple strategies:
      - ORB (Opening Range Breakout)
      - Buy and Hold
      - Moving Average Crossover
      - Mean Reversion
      - Volatility Breakout
      - Intraday Momentum
      - Golden Ratio Breakouts

    Assumes the DataFrame 'df' contains required columns:
      - "<close>", "<open>", "<high>", "<low>"
      - For ORB, the ORB strategy signals should be in "signal" and 
        opening range info in "opening_range_high", "opening_range_low", and "range_size".
    
    Returns:
      - dict: Keys are strategy names, values are cumulative return pd.Series.
    """
    df = df.copy()
    
    # Calculate daily returns (or intraday returns for high frequency data)
    df['daily_returns'] = df['<close>'].pct_change()
    
    results = {}

    # --- ORB Strategy ---
    # Shift the ORB signal (assumed to be in 'signal') to simulate taking the next bar's position
    df['position_orb'] = df['signal'].shift()
    df['strategy_returns_orb'] = df['position_orb'] * df['daily_returns']
    cumulative_returns_orb = (1 + df['strategy_returns_orb']).cumprod() - 1
    results['ORB'] = cumulative_returns_orb
    logging.info("Backtested ORB strategy.")

    # --- Buy and Hold Strategy ---
    df['position_bh'] = 1
    df['strategy_returns_bh'] = df['position_bh'] * df['daily_returns']
    cumulative_returns_bh = (1 + df['strategy_returns_bh']).cumprod() - 1
    results['Buy and Hold'] = cumulative_returns_bh
    logging.info("Backtested Buy and Hold strategy.")

    # --- Moving Average Crossover Strategy ---
    df['short_MA'] = df['<close>'].rolling(window=20, min_periods=1).mean()
    df['long_MA'] = df['<close>'].rolling(window=50, min_periods=1).mean()
    df['signal_ma'] = np.where(df['short_MA'] > df['long_MA'], 1, -1)
    df['position_ma'] = df['signal_ma'].shift()
    df['strategy_returns_ma'] = df['position_ma'] * df['daily_returns']
    cumulative_returns_ma = (1 + df['strategy_returns_ma']).cumprod() - 1
    results['MA Crossover'] = cumulative_returns_ma
    logging.info("Backtested Moving Average Crossover strategy.")

    # --- Mean Reversion Strategy ---
    window = 20
    df['rolling_mean'] = df['<close>'].rolling(window=window, min_periods=1).mean()
    df['rolling_std'] = df['<close>'].rolling(window=window, min_periods=1).std()
    df['z_score'] = (df['<close>'] - df['rolling_mean']) / df['rolling_std']
    threshold = 1.0  # can be adjusted
    df['signal_mr'] = 0
    df.loc[df['z_score'] > threshold, 'signal_mr'] = -1  # Overbought: signal short
    df.loc[df['z_score'] < -threshold, 'signal_mr'] = 1  # Oversold: signal long
    df['position_mr'] = df['signal_mr'].shift()
    df['strategy_returns_mr'] = df['position_mr'] * df['daily_returns']
    cumulative_returns_mr = (1 + df['strategy_returns_mr']).cumprod() - 1
    results['Mean Reversion'] = cumulative_returns_mr
    logging.info("Backtested Mean Reversion strategy.")

    # --- Volatility Breakout Strategy ---
    # Simple rule: if close > open + threshold*(high-low), then long; if close < open - threshold*(high-low), then short.
    vol_breakout_threshold = 0.5  # parameter can be adjusted
    df['signal_vb'] = 0
    df.loc[df['<close>'] > df['<open>'] + vol_breakout_threshold * (df['<high>'] - df['<low>']), 'signal_vb'] = 1
    df.loc[df['<close>'] < df['<open>'] - vol_breakout_threshold * (df['<high>'] - df['<low>']), 'signal_vb'] = -1
    df['position_vb'] = df['signal_vb'].shift()
    df['strategy_returns_vb'] = df['position_vb'] * df['daily_returns']
    cumulative_returns_vb = (1 + df['strategy_returns_vb']).cumprod() - 1
    results['Volatility Breakout'] = cumulative_returns_vb
    logging.info("Backtested Volatility Breakout strategy.")

    # --- Intraday Momentum Strategy ---
    # Signal based on the intraday price change from open.
    df['momentum'] = df['<close>'] - df['<open>']
    df['signal_mom'] = np.where(df['momentum'] > 0, 1, -1)
    df['position_mom'] = df['signal_mom'].shift()
    df['strategy_returns_mom'] = df['position_mom'] * df['daily_returns']
    cumulative_returns_mom = (1 + df['strategy_returns_mom']).cumprod() - 1
    results['Intraday Momentum'] = cumulative_returns_mom
    logging.info("Backtested Intraday Momentum strategy.")

    # --- Golden Ratio Breakouts Strategy ---
    # Assumes the ORB opening range has been computed.
    # Ensure that "opening_range_low" and "opening_range_high" are available.
    # Compute range_size if not already present.
    if 'range_size' not in df.columns:
        df['range_size'] = df['opening_range_high'] - df['opening_range_low']
    golden_ratio = 0.618
    df['fibo_level_long'] = df['opening_range_low'] + golden_ratio * df['range_size']
    df['fibo_level_short'] = df['opening_range_low'] + (1 - golden_ratio) * df['range_size']  # approx. 0.382 level
    df['signal_gr'] = 0
    df.loc[df['<close>'] > df['fibo_level_long'], 'signal_gr'] = 1
    df.loc[df['<close>'] < df['fibo_level_short'], 'signal_gr'] = -1
    df['position_gr'] = df['signal_gr'].shift()
    df['strategy_returns_gr'] = df['position_gr'] * df['daily_returns']
    cumulative_returns_gr = (1 + df['strategy_returns_gr']).cumprod() - 1
    results['Golden Ratio Breakout'] = cumulative_returns_gr
    logging.info("Backtested Golden Ratio Breakout strategy.")

    return results


# --- Performance Metrics Calculation ---
def compute_performance_metrics(returns_series):
    """
    Compute performance metrics:
      - Net Return: Final cumulative return.
      - Sharpe Ratio: (Mean return / Std return) annualized.
      - Sortino Ratio: Mean return divided by the downside std, annualized.
      - Maximum Drawdown.
      - Win Rate: Fraction of non-zero return bars that are positive.
      - Average Trade Return.
      - Win/Loss Ratio.
    """
    net_return = returns_series.iloc[-1]
    mean_return = returns_series.mean()
    std_return = returns_series.std()
    sharpe_ratio = (mean_return / std_return) * np.sqrt(252) if std_return != 0 else np.nan
    negative_returns = returns_series[returns_series < 0]
    std_downside = negative_returns.std()
    sortino_ratio = (mean_return / std_downside) * np.sqrt(252) if std_downside != 0 else np.nan
    cumulative = (1 + returns_series).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()
    trades = returns_series[returns_series != 0]
    win_rate = (trades > 0).mean() if len(trades) > 0 else np.nan
    avg_trade_return = trades.mean() if len(trades) > 0 else np.nan
    wins = trades[trades > 0]
    losses = trades[trades < 0]
    avg_win = wins.mean() if len(wins) > 0 else 0
    avg_loss = losses.mean() if len(losses) > 0 else 0
    win_loss_ratio = avg_win / abs(avg_loss) if avg_loss != 0 else np.nan
    return {
        'net_return': net_return,
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'avg_trade_return': avg_trade_return,
        'win_loss_ratio': win_loss_ratio
    }


# --- Performance Comparison Plotting ---
def plot_performance_comparison(metrics_dict):
    """
    Plot a grouped bar chart comparing key performance metrics across strategies.
    
    Parameters:
      - metrics_dict: dict with keys as strategy names and values as dictionaries of metrics.
    """
    metrics_df = pd.DataFrame(metrics_dict).T
    # Select key metrics for plotting
    metrics_df = metrics_df[['net_return', 'sharpe_ratio', 'sortino_ratio', 'max_drawdown']]
    metrics_df.plot(kind='bar', figsize=(12, 6))
    plt.title("Performance Comparison Across Strategies")
    plt.ylabel("Metric Value")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# --- Parameter Optimization with Performance Evaluation ---
def optimize_parameters_gs(rangewise_results, criteria_sharpe_threshold, criteria_max_drawdown):
    """
    Run the backtesting engine over a grid of parameter combinations, compute performance
    metrics for the ORB strategy, and select the best parameter combination based on sample criteria.

    Sample Criteria:
      - Sharpe ratio must be at least criteria_sharpe_threshold.
      - Maximum drawdown must not be worse than criteria_max_drawdown.
      - Among the valid combinations, choose the one with the highest Sharpe ratio.

    Returns:
      - performance_df (pd.DataFrame): Metrics for all parameter combinations.
      - best_parameters (str): Key string for the best parameter combination (or None if no combination qualifies).
    """
    performance_records = []
    for key, df in rangewise_results.items():
        strat_returns = backtest_strategies(df)
        orb_returns = strat_returns.get('ORB')
        metrics = compute_performance_metrics(orb_returns)
        record = {
            'parameters': key,
            'net_return': metrics['net_return'],
            'sharpe_ratio': metrics['sharpe_ratio'],
            'sortino_ratio': metrics['sortino_ratio'],
            'max_drawdown': metrics['max_drawdown'],
            'win_rate': metrics['win_rate'],
            'avg_trade_return': metrics['avg_trade_return'],
            'win_loss_ratio': metrics['win_loss_ratio']
        }
        performance_records.append(record)
    performance_df = pd.DataFrame(performance_records)

    print(performance_df)
    
    valid_df = performance_df[(performance_df['sharpe_ratio'] >= criteria_sharpe_threshold) &
                              (performance_df['max_drawdown'] >= criteria_max_drawdown)]
    print(valid_df)
    if not valid_df.empty:
        best_row = valid_df.sort_values(by='sharpe_ratio', ascending=False).iloc[0]
        best_parameters = best_row['parameters']
        logging.info(f"Best parameters: {best_parameters}")
    else:
        best_parameters = None
        logging.info("No parameter combination met the criteria.")

    return performance_df, best_parameters




# ------------------------------
# Evolutionary Algorithm Functions
# ------------------------------
def evaluate_candidate(data, candidate):
    """
    Evaluate a candidate parameter set by running the ORB strategy and backtesting.
    
    Returns a tuple (fitness, metrics) where fitness is defined as the Sharpe ratio
    if max drawdown is acceptable and net return is positive; otherwise, penalize the candidate.
    """
    # Compute ORB strategy using candidate parameters.
    data_or = calculate_opening_range(data.copy(), candidate["opening_range_minutes"], candidate["market_open_time"])
    df_orb = apply_orb_strategy(data_or.copy(), profit_target_multiplier=candidate["profit_target_multiplier"], stop_loss_multiplier=candidate["stop_loss_multiplier"])
    strat_returns = backtest_strategies(df_orb)
    orb_returns = strat_returns.get("ORB")
    metrics = compute_performance_metrics(orb_returns)
    
    # Define fitness: penalize if net return is negative or max drawdown is worse than -10%
    if metrics["max_drawdown"] < -0.1 or metrics["net_return"] < 0:
        fitness = -1000
    else:
        fitness = metrics["sharpe_ratio"]
    return fitness, metrics




def evolve_parameters(data, pop_size=20, generations=20):
    """
    Evolve candidate parameter sets using a simple genetic algorithm.

    Search space:
      - profit_target_multiplier: continuous in [1.5, 2.5]
      - stop_loss_multiplier: continuous in [0.5, 1.5]
      - opening_range_minutes: discrete from [15, 30]
      - market_open_time: discrete from ["09:15:00", "09:30:00"]

    Returns the best candidate (dict) and its fitness.
    """
    profit_target_range = (1.5, 2.5)
    stop_loss_range = (0.5, 1.5)
    opening_range_options = [15, 30]
    market_open_options = ["09:15:00", "09:30:00"]

    # Initialize random population
    population = []
    for _ in range(pop_size):
        candidate = {
            "profit_target_multiplier": random.uniform(*profit_target_range),
            "stop_loss_multiplier": random.uniform(*stop_loss_range),
            "opening_range_minutes": random.choice(opening_range_options),
            "market_open_time": random.choice(market_open_options)
        }
        population.append(candidate)

    best_candidate = None
    best_fitness = -float("inf")
    
    for gen in range(generations):
        evaluated = []
        for candidate in population:
            fitness, metrics = evaluate_candidate(data, candidate)
            evaluated.append((candidate, fitness, metrics))
        evaluated.sort(key=lambda x: x[1], reverse=True)
        best_in_gen = evaluated[0]
        if best_in_gen[1] > best_fitness:
            best_fitness = best_in_gen[1]
            best_candidate = best_in_gen[0]
        logging.info(f"Generation {gen+1}: Best fitness = {best_in_gen[1]:.4f}, Candidate: {best_in_gen[0]}")
        
        # Select top 50% as parents
        num_parents = pop_size // 2
        parents = [cand for cand, fit, met in evaluated[:num_parents]]
        
        # Create new population via crossover and mutation
        new_population = []
        while len(new_population) < pop_size:
            parent1 = random.choice(parents)
            parent2 = random.choice(parents)
            # Crossover: average continuous parameters and randomly choose discrete ones.
            child = {
                "profit_target_multiplier": (parent1["profit_target_multiplier"] + parent2["profit_target_multiplier"]) / 2,
                "stop_loss_multiplier": (parent1["stop_loss_multiplier"] + parent2["stop_loss_multiplier"]) / 2,
                "opening_range_minutes": random.choice([parent1["opening_range_minutes"], parent2["opening_range_minutes"]]),
                "market_open_time": random.choice([parent1["market_open_time"], parent2["market_open_time"]])
            }
            # Mutation: with a small chance, slightly adjust continuous parameters.
            mutation_rate = 0.1
            if random.random() < mutation_rate:
                child["profit_target_multiplier"] += random.uniform(-0.1, 0.1)
                child["profit_target_multiplier"] = max(profit_target_range[0], min(profit_target_range[1], child["profit_target_multiplier"]))
            if random.random() < mutation_rate:
                child["stop_loss_multiplier"] += random.uniform(-0.1, 0.1)
                child["stop_loss_multiplier"] = max(stop_loss_range[0], min(stop_loss_range[1], child["stop_loss_multiplier"]))
            if random.random() < mutation_rate:
                child["opening_range_minutes"] = random.choice(opening_range_options)
            if random.random() < mutation_rate:
                child["market_open_time"] = random.choice(market_open_options)
            new_population.append(child)
        population = new_population

    return best_candidate, best_fitness






'''

# --- --------- Calculate Performance Metrics --------- ---
def calculate_performance_metrics(cumulative_returns, daily_returns):
    """
    Calculate key performance metrics for a strategy.

    Parameters:
    - cumulative_returns (pd.Series): Cumulative returns of the strategy.
    - daily_returns (pd.Series): Daily returns of the strategy.

    Returns:
    - dict: Dictionary containing performance metrics.
    """
    total_return = cumulative_returns.iloc[-1]

    num_days = (cumulative_returns.index[-1] - cumulative_returns.index[0]).days
    num_years = num_days / 365.25
    annualized_return = (1 + total_return) ** (1 / num_years) - 1

    # Calculate drawdowns
    rolling_max = (1 + cumulative_returns).cummax()
    drawdowns = (1 + cumulative_returns) / rolling_max - 1
    max_drawdown = drawdowns.min()

    # Calculate Sharpe Ratio (Assuming risk-free rate = 0)
    if daily_returns.std() != 0:
        sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)  # 252 trading days
    else:
        sharpe_ratio = np.nan  # Avoid division by zero

    return {
        'Total Return': total_return,
        'Annualized Return': annualized_return,
        'Maximum Drawdown': max_drawdown,
        'Sharpe Ratio': sharpe_ratio
    }


# --- --------- Plotting Functions --------- ---
def plot_cumulative_returns(cumulative_returns_orb, cumulative_returns_bh, output_dir, ticker):
    """
    Plot and save cumulative returns for ORB and Buy and Hold strategies.

    Parameters:
    - cumulative_returns_orb (pd.Series): Cumulative returns for ORB.
    - cumulative_returns_bh (pd.Series): Cumulative returns for Buy and Hold.
    - output_dir (Path): Directory to save the plot.
    """
    plt.figure(figsize=(14, 7))
    plt.plot(cumulative_returns_orb, label='ORB Strategy Returns', color='blue')
    plt.plot(cumulative_returns_bh, label='Buy and Hold Returns', color='orange', linestyle='--')
    plt.title('Opening Range Breakout vs. Buy and Hold Strategy Cumulative Returns (2019-2024)')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Returns')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Define the path to save the cumulative returns plot
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    cumulative_returns_plot_dir = output_dir / ticker
    cumulative_returns_plot_path = cumulative_returns_plot_dir / f'cumulative_returns_orb_vs_bh_{timestamp}.png'
    try:
        cumulative_returns_plot_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output directory created at: {cumulative_returns_plot_dir.resolve()}")
    except Exception as e:
        print(f"Failed to create output directory at {cumulative_returns_plot_dir}: {e}")
        raise e
    
    # Save the plot
    plt.savefig(cumulative_returns_plot_path)
    logging.info(f"Cumulative Returns plot saved to {cumulative_returns_plot_path}")

    # Display the plot
    # plt.show()

    plt.close()


def plot_drawdowns(cumulative_returns_orb, cumulative_returns_bh, output_dir, ticker):
    """
    Plot and save drawdowns for ORB and Buy and Hold strategies.

    Parameters:
    - cumulative_returns_orb (pd.Series): Cumulative returns for ORB.
    - cumulative_returns_bh (pd.Series): Cumulative returns for Buy and Hold.
    - output_dir (Path): Directory to save the plot.
    """
    # Calculate rolling maximum
    rolling_max_orb = (1 + cumulative_returns_orb).cummax()
    rolling_max_bh = (1 + cumulative_returns_bh).cummax()

    # Calculate drawdowns
    drawdowns_orb = (1 + cumulative_returns_orb) / rolling_max_orb - 1
    drawdowns_bh = (1 + cumulative_returns_bh) / rolling_max_bh - 1

    # Plot drawdowns
    plt.figure(figsize=(14, 7))
    plt.plot(drawdowns_orb, label='ORB Strategy Drawdown', color='blue')
    plt.plot(drawdowns_bh, label='Buy and Hold Drawdown', color='orange', linestyle='--')
    plt.title('Drawdowns: ORB Strategy vs. Buy and Hold (2019-2024)')
    plt.xlabel('Date')
    plt.ylabel('Drawdown')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Define the path to save the drawdowns plot
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    drawdowns_plot_dir = output_dir / ticker
    drawdowns_plot_path = drawdowns_plot_dir / f'drawdowns_orb_vs_bh_{timestamp}.png'

    try:
        drawdowns_plot_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output directory created at: {drawdowns_plot_dir.resolve()}")
    except Exception as e:
        print(f"Failed to create output directory at {drawdowns_plot_dir}: {e}")
        raise e

    # Save the plot
    plt.savefig(drawdowns_plot_path)
    logging.info(f"Drawdowns plot saved to {drawdowns_plot_path}")

    # Display the plot
    # plt.show()

    plt.close()


def plot_performance_metrics(performance_df, output_dir, ticker):
    """
    Plot and save performance metrics comparison as a bar chart.

    Parameters:
    - performance_df (pd.DataFrame): DataFrame containing performance metrics.
    - output_dir (Path): Directory to save the plot.
    """
    # Prepare data for plotting
    performance_plot = performance_df.drop(['Maximum Drawdown'], axis=0).T * 100  # Convert to percentages

    # Plotting
    plt.figure(figsize=(10, 6))
    performance_plot.plot(kind='bar', figsize=(10, 6))
    plt.title('Performance Metrics Comparison: ORB Strategy vs. Buy and Hold (2019-2024)')
    plt.ylabel('Value (%)')
    plt.xlabel('Strategy')
    plt.xticks(rotation=0)
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter())
    plt.legend(loc='upper left')
    plt.grid(axis='y')
    plt.tight_layout()

    # Define the path to save the performance metrics plot
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    performance_metrics_plot_dir = output_dir / ticker
    performance_metrics_plot_path = performance_metrics_plot_dir / f'performance_metrics_comparison_{timestamp}.png'

    try:
        performance_metrics_plot_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output directory created at: {performance_metrics_plot_dir.resolve()}")
    except Exception as e:
        print(f"Failed to create output directory at {performance_metrics_plot_dir}: {e}")
        raise e

    # Save the plot
    plt.savefig(performance_metrics_plot_path)
    logging.info(f"Performance metrics plot saved to {performance_metrics_plot_path}")

    # Display the plot
    # plt.show()

    plt.close()


# --- --------- Save Performance Metrics --------- ---
def save_performance_metrics(performance_df, output_dir, ticker):
    """
    Save the performance metrics DataFrame to a CSV file.

    Parameters:
    - performance_df (pd.DataFrame): DataFrame containing performance metrics.
    - output_dir (Path): Directory to save the CSV file.
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    performance_metrics_dir = output_dir / ticker
    performance_metrics_path = performance_metrics_dir / f'performance_metrics_comparison_{timestamp}.csv'

    try:
        performance_metrics_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output directory created at: {performance_metrics_dir.resolve()}")
    except Exception as e:
        print(f"Failed to create output directory at {performance_metrics_dir}: {e}")
        raise e

    performance_df.to_csv(performance_metrics_path)
    logging.info(f"Performance metrics saved to {performance_metrics_path}")


# --- --------- Save Enhanced Dataset --------- ---
def save_enhanced_dataset(df, output_dir, ticker):
    """
    Save the enhanced DataFrame with strategy metrics to a CSV file.

    Parameters:
    - df (pd.DataFrame): Enhanced DataFrame.
    - output_dir (Path): Directory to save the CSV file.
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    enhanced_dataset_dir = output_dir / ticker
    output_file = enhanced_dataset_dir / f'enhanced_orb_strategy_{timestamp}.csv'

    try:
        enhanced_dataset_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output directory created at: {enhanced_dataset_dir.resolve()}")
    except Exception as e:
        print(f"Failed to create output directory at {enhanced_dataset_dir}: {e}")
        raise e

    df.to_csv(output_file, index=True)
    logging.info(f"Enhanced dataset with ORB and B&H strategy metrics saved to {output_file}")


'''
