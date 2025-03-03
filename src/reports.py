import matplotlib.pyplot as plt
import pandas as pd


def plot_returns_comparison(bt_df, ticker, best_parameters, save_plot=False, filename=None):
    """
    Plot a single chart that overlays three types of returns for each strategy:
      - Cumulative Returns (line chart)
      - Running Net Returns (cumulative - 1)
      - Rolling Annualized Returns (computed as (cum_value)^(252/(t+1)) - 1)
      
    The chart title includes the ticker and best_parameters.
    
    Parameters:
      - bt_df: Backtested dataframe containing cumulative returns columns.
      - ticker: Ticker string.
      - best_parameters: Best parameter combination string.
      - save_plot: bool, if True the plot is saved to file.
      - filename: str, filename to save the plot.
    """
    strategy_cols = {
        "ORB": "ORB_cum",
        "Buy and Hold": "BH_cum",
        "MA Crossover": "MA_cum",
        "Mean Reversion": "MR_cum",
        "Volatility Breakout": "VB_cum",
        "Intraday Momentum": "MOM_cum",
        "Golden Ratio Breakout": "GR_cum"
    }
    
    # Prepare a figure that overlays all three return types for each strategy
    fig, ax = plt.subplots(figsize=(14, 8))
    
    for strat, col in strategy_cols.items():
        cum_series = bt_df[col]
        # Running Net Returns: cumulative - 1
        net_series = cum_series - 1
        # Rolling Annualized Returns: computed for each time index t (starting at 1)
        roll_ann = [ (val ** (252 / (i+1)) - 1) for i, val in enumerate(cum_series) ]
        ax.plot(bt_df.index, cum_series, label=f"{strat} Cum")
        ax.plot(bt_df.index, net_series, linestyle="--", label=f"{strat} Net")
        ax.plot(bt_df.index, roll_ann, linestyle=":", label=f"{strat} Ann")
    
    ax.set_title(f"All Returns for {ticker} | Best Params - {best_parameters}", fontsize=18)
    ax.set_xlabel("Trading Days")
    ax.set_ylabel("Return")
    ax.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    # plt.show()
    if save_plot and filename:
        plt.savefig(filename)
    # plt.show()




def plot_performance_comparison(metrics_dict, ticker, best_parameters, save_plot=False, filename=None):
    """
    Plot performance metrics across strategies in 3 subplots:
      1. Return Metrics: net_return and annualized_return.
      2. Risk Metrics: annualized_volatility, max_drawdown, and calmar_ratio.
      3. Other Ratios: sharpe_ratio, adjusted_sharpe_ratio, omega_ratio, kappa_ratio, sortino_ratio, treynor_ratio,
         win_rate, avg_trade_return, win_loss_ratio.
    
    Parameters:
      - metrics_dict: dict with keys as strategy names and values as dicts of metrics.
      - save_plot: bool, if True the plot is saved to file.
      - filename: str, filename to save the plot.
    """
    metrics_df = pd.DataFrame(metrics_dict).T

    cat1 = ["annualized_volatility", "max_drawdown", "calmar_ratio"]
    cat2 = ["sharpe_ratio", "adjusted_sharpe_ratio", "omega_ratio", "kappa_ratio", "sortino_ratio", "treynor_ratio"]
    cat3 = ["win_rate", "avg_trade_return", "win_loss_ratio"]
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 18))
    
    metrics_df[cat1].plot(kind="bar", ax=axes[0])
    axes[0].set_title("Risk Metrics")
    axes[0].set_ylabel("Value")
    axes[0].tick_params(axis="x", rotation=45)
    
    metrics_df[cat2].plot(kind="bar", ax=axes[1])
    axes[1].set_title("Performance Metrics")
    axes[1].set_ylabel("Value")
    axes[1].tick_params(axis="x", rotation=45)
    
    metrics_df[cat3].plot(kind="bar", ax=axes[2])
    axes[2].set_title("Trade-level Ratios")
    axes[2].set_ylabel("Value")
    axes[2].tick_params(axis="x", rotation=45)
    
    plt.suptitle(f"Performance Metrics Comparison Across Strategies for {ticker} | Best Params - {best_parameters}", fontsize=18)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    # plt.show()
    
    if save_plot and filename:
        plt.savefig(filename)
    # plt.show()