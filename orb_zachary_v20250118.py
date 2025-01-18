import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import glob
from pathlib import Path
import os
import logging
from datetime import datetime

# --- --------- Configuration Parameters --------- ---
# Parameters for the Opening Range Breakout strategy
OPENING_RANGE_MINUTES = 15  # Duration of the opening range in minutes (e.g., first 15 minutes)
PROFIT_TARGET_MULTIPLIER = 2  # Multiplier for setting profit targets
STOP_LOSS_MULTIPLIER = 1  # Multiplier for setting stop-loss levels

# Paths
DATA_DIR = Path('data')  # Directory containing CSV files
OUTPUT_DIR = DATA_DIR / 'output'  # Directory to save outputs

# Strategy Parameters
MARKET_OPEN_TIME = '09:30:00'  # Market opening time


# --- --------- Create Output Directory --------- ---
def create_output_directory(output_dir):
    """
    Create the output directory if it doesn't exist.

    Parameters:
    - output_dir (Path): Path to the output directory.
    """
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output directory created at: {output_dir.resolve()}")
    except Exception as e:
        print(f"Failed to create output directory at {output_dir}: {e}")
        raise e


# --- --------- Setup Logging --------- ---
def setup_logging(output_dir):
    """
    Configure logging to capture the script's execution details.

    Parameters:
    - output_dir (Path): Directory where the log file will be saved.
    """
    log_file = output_dir / 'orb_backtest.log'
    logging.basicConfig(
        filename=log_file,
        filemode='a',
        format='%(asctime)s %(levelname)s:%(message)s',
        level=logging.INFO
    )
    logging.info("Logging has been configured.")


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
        opening_range = data.groupby('date').apply(
            lambda x: x[x.index.time <= threshold_time]
        ).reset_index(level=0, drop=True)
        logging.info(f"Total records within the opening range: {len(opening_range)}")
    except FutureWarning as fw:
        logging.warning(f"FutureWarning encountered: {fw}")
        # Handle the warning by modifying the groupby apply as needed
        opening_range = data.groupby('date').filter(
            lambda x: (x.index.time <= threshold_time).any()
        ).copy()
        opening_range = opening_range.groupby('date').apply(
            lambda x: x[x.index.time <= threshold_time]
        ).reset_index(drop=True)
        logging.info(f"Total records within the opening range after handling warning: {len(opening_range)}")

    # Calculate opening range high and low
    opening_range_high = opening_range.groupby('date')['high'].max()
    opening_range_low = opening_range.groupby('date')['low'].min()

    # Merge back into main data
    data['opening_range_high'] = data['date'].map(opening_range_high)
    data['opening_range_low'] = data['date'].map(opening_range_low)
    logging.info("Merged opening range high and low back into the main DataFrame.")

    return data


# --- --------- Define ORB Strategy --------- ---
def apply_orb_strategy(df, profit_target_multiplier=2, stop_loss_multiplier=1):
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
    df['long_entry'] = df['close'] > df['opening_range_high']
    df['short_entry'] = df['close'] < df['opening_range_low']
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

    return df


# --- --------- Backtest Strategies --------- ---
def backtest_strategies(df):
    """
    Backtest ORB and Buy and Hold strategies.

    Parameters:
    - df (pd.DataFrame): DataFrame with strategy signals.

    Returns:
    - pd.Series, pd.Series: Cumulative returns for ORB and Buy and Hold strategies.
    """
    # --- Opening Range Breakout Strategy ---

    # Shift signals to represent positions taken at the next time step
    df['position_orb'] = df['signal'].shift()
    logging.info("Shifted ORB signals to represent next time step positions.")

    # Calculate daily returns based on closing prices
    df['daily_returns'] = df['close'].pct_change()

    # Strategy returns: position * daily returns
    df['strategy_returns_orb'] = df['position_orb'] * df['daily_returns']
    logging.info("Calculated strategy returns for ORB.")

    # Calculate cumulative returns for ORB
    cumulative_returns_orb = (1 + df['strategy_returns_orb']).cumprod() - 1
    logging.info("Calculated cumulative returns for ORB.")

    # --- Buy and Hold Strategy ---

    # Define Buy and Hold position (always 1)
    df['position_bh'] = 1
    logging.info("Defined Buy and Hold position.")

    # Strategy returns for Buy and Hold
    df['strategy_returns_bh'] = df['position_bh'] * df['daily_returns']
    logging.info("Calculated strategy returns for Buy and Hold.")

    # Calculate cumulative returns for Buy and Hold
    cumulative_returns_bh = (1 + df['strategy_returns_bh']).cumprod() - 1
    logging.info("Calculated cumulative returns for Buy and Hold.")

    return cumulative_returns_orb, cumulative_returns_bh


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
def plot_cumulative_returns(cumulative_returns_orb, cumulative_returns_bh, output_dir):
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
    cumulative_returns_plot_path = output_dir / f'cumulative_returns_orb_vs_bh_{timestamp}.png'

    # Save the plot
    plt.savefig(cumulative_returns_plot_path)
    logging.info(f"Cumulative Returns plot saved to {cumulative_returns_plot_path}")

    # Display the plot
    plt.show()


def plot_drawdowns(cumulative_returns_orb, cumulative_returns_bh, output_dir):
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
    drawdowns_plot_path = output_dir / f'drawdowns_orb_vs_bh_{timestamp}.png'

    # Save the plot
    plt.savefig(drawdowns_plot_path)
    logging.info(f"Drawdowns plot saved to {drawdowns_plot_path}")

    # Display the plot
    plt.show()


def plot_performance_metrics(performance_df, output_dir):
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
    performance_metrics_plot_path = output_dir / f'performance_metrics_comparison_{timestamp}.png'

    # Save the plot
    plt.savefig(performance_metrics_plot_path)
    logging.info(f"Performance metrics plot saved to {performance_metrics_plot_path}")

    # Display the plot
    plt.show()


# --- --------- Save Performance Metrics --------- ---
def save_performance_metrics(performance_df, output_dir):
    """
    Save the performance metrics DataFrame to a CSV file.

    Parameters:
    - performance_df (pd.DataFrame): DataFrame containing performance metrics.
    - output_dir (Path): Directory to save the CSV file.
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    performance_metrics_path = output_dir / f'performance_metrics_comparison_{timestamp}.csv'
    performance_df.to_csv(performance_metrics_path)
    logging.info(f"Performance metrics saved to {performance_metrics_path}")


# --- --------- Save Enhanced Dataset --------- ---
def save_enhanced_dataset(df, output_dir):
    """
    Save the enhanced DataFrame with strategy metrics to a CSV file.

    Parameters:
    - df (pd.DataFrame): Enhanced DataFrame.
    - output_dir (Path): Directory to save the CSV file.
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = output_dir / f'enhanced_orb_strategy_{timestamp}.csv'
    df.to_csv(output_file, index=True)
    logging.info(f"Enhanced dataset with ORB and B&H strategy metrics saved to {output_file}")


# --- --------- Main Execution Flow --------- ---
def main():
    try:
        # Step 1: Create Output Directory
        create_output_directory(OUTPUT_DIR)

        # Step 2: Setup Logging
        setup_logging(OUTPUT_DIR)
        logging.info("ORB Strategy Backtest Script Started.")

        # Step 3: Load and Prepare Data
        data = load_and_concatenate_csv(DATA_DIR)
        data = prepare_data(data)

        # Step 4: Calculate Opening Range
        data = calculate_opening_range(data, OPENING_RANGE_MINUTES, MARKET_OPEN_TIME)

        # Step 5: Apply ORB Strategy
        data = apply_orb_strategy(data, PROFIT_TARGET_MULTIPLIER, STOP_LOSS_MULTIPLIER)

        # Step 6: Backtest Strategies
        cumulative_returns_orb, cumulative_returns_bh = backtest_strategies(data)
        logging.info("Completed backtesting of both ORB and Buy and Hold strategies.")

        # Step 7: Calculate Performance Metrics
        daily_returns_orb = data['strategy_returns_orb'].dropna()
        daily_returns_bh = data['strategy_returns_bh'].dropna()

        metrics_orb = calculate_performance_metrics(cumulative_returns_orb.dropna(), daily_returns_orb)
        metrics_bh = calculate_performance_metrics(cumulative_returns_bh.dropna(), daily_returns_bh)

        # Compile performance metrics into DataFrame
        performance_df = pd.DataFrame({
            'ORB Strategy': metrics_orb,
            'Buy and Hold': metrics_bh
        })

        logging.info("Calculated performance metrics for both strategies.")
        print("\nPerformance Metrics Comparison:")
        print(performance_df)

        # Step 8: Plotting
        plot_cumulative_returns(cumulative_returns_orb, cumulative_returns_bh, OUTPUT_DIR)
        plot_drawdowns(cumulative_returns_orb, cumulative_returns_bh, OUTPUT_DIR)
        plot_performance_metrics(performance_df, OUTPUT_DIR)

        # Step 9: Save Performance Metrics to CSV
        save_performance_metrics(performance_df, OUTPUT_DIR)

        # Step 10: Save the Enhanced Dataset
        save_enhanced_dataset(data, OUTPUT_DIR)

        logging.info("ORB Strategy Backtest Script completed successfully.")

    except Exception as e:
        logging.error(f"An error occurred during execution: {e}")
        print(f"An error occurred: {e}")


# Execute the main function
if __name__ == "__main__":
    main()