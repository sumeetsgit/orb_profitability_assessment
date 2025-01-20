import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Parameters for the Opening Range Breakout strategy
opening_range_minutes = 15  # Define the opening range time window (e.g., 15 minutes)
profit_target_multiplier = 2  # Take-profit multiplier based on the opening range size
stop_loss_multiplier = 1  # Stop-loss multiplier based on the opening range size

# Load your dataset
# The dataset should have columns: ['datetime', 'open', 'high', 'low', 'close', 'volume']
data = pd.read_csv('your_data.csv', parse_dates=['datetime'])
data.set_index('datetime', inplace=True)

# Add a date column for grouping
data['date'] = data.index.date

# Filter data to include only the first X minutes of each trading day
opening_range = data.groupby('date').apply(lambda x: x[x.index.time <= pd.Timestamp('09:30:00').time() + pd.Timedelta(minutes=opening_range_minutes)])

# Calculate opening range high and low
opening_range_high = opening_range.groupby('date')['high'].max()
opening_range_low = opening_range.groupby('date')['low'].min()

# Merge opening range values back into the main dataset
data['opening_range_high'] = data['date'].map(opening_range_high)
data['opening_range_low'] = data['date'].map(opening_range_low)

def apply_orb_strategy(data, profit_target_multiplier, stop_loss_multiplier):
    """
    Apply Opening Range Breakout Strategy.
    """
    data['long_entry'] = data['close'] > data['opening_range_high']
    data['short_entry'] = data['close'] < data['opening_range_low']

    # Calculate the range size
    data['range_size'] = data['opening_range_high'] - data['opening_range_low']

    # Define profit target and stop loss levels
    data['profit_target_long'] = data['opening_range_high'] + profit_target_multiplier * data['range_size']
    data['stop_loss_long'] = data['opening_range_high'] - stop_loss_multiplier * data['range_size']
    data['profit_target_short'] = data['opening_range_low'] - profit_target_multiplier * data['range_size']
    data['stop_loss_short'] = data['opening_range_low'] + stop_loss_multiplier * data['range_size']

    # Initialize signals
    data['signal'] = 0

    # Generate signals for long and short positions
    data.loc[data['long_entry'], 'signal'] = 1  # Buy signal
    data.loc[data['short_entry'], 'signal'] = -1  # Sell signal

    return data

# Apply the strategy to the dataset
data = apply_orb_strategy(data, profit_target_multiplier, stop_loss_multiplier)

# Backtesting the strategy
def backtest_strategy(data):
    """
    Backtest the strategy to calculate P&L.
    """
    data['position'] = data['signal'].shift()  # Lag the signal to avoid lookahead bias
    data['returns'] = data['close'].pct_change()  # Calculate daily returns
    data['strategy_returns'] = data['position'] * data['returns']  # Apply strategy returns

    cumulative_returns = (1 + data['strategy_returns']).cumprod() - 1

    return cumulative_returns

# Run backtest
cumulative_returns = backtest_strategy(data)

# Plot the cumulative returns
plt.figure(figsize=(12, 6))
plt.plot(cumulative_returns, label='Strategy Returns')
plt.title('Opening Range Breakout Strategy Cumulative Returns')
plt.xlabel('Date')
plt.ylabel('Cumulative Returns')
plt.legend()
plt.show()

# Save the enhanced dataset with signals and strategy metrics
data.to_csv('enhanced_orb_strategy.csv', index=True)
