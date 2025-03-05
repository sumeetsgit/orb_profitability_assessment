import numpy as np
from settings import RISK_FREE_RATE


def calculate_max_drawdown(cum_series):
    """
    Calculate the maximum drawdown from a cumulative portfolio value series.
    """
    running_max = cum_series.cummax()
    drawdown = (cum_series - running_max) / running_max
    return drawdown.min()


def calculate_win_rate(daily_returns):
    """
    Calculate the win rate based on non-zero trade returns.
    """
    trades = daily_returns[daily_returns != 0]
    return (trades > 0).mean() if len(trades) > 0 else np.nan


def calculate_avg_trade_return(daily_returns):
    """
    Calculate the average trade return.
    """
    trades = daily_returns[daily_returns != 0]
    return trades.mean() if len(trades) > 0 else np.nan


def calculate_win_loss_ratio(daily_returns):
    """
    Calculate the win/loss ratio based on average winning and losing trade returns.
    """
    trades = daily_returns[daily_returns != 0]
    wins = trades[trades > 0]
    losses = trades[trades < 0]
    avg_win = wins.mean() if len(wins) > 0 else 0
    avg_loss = losses.mean() if len(losses) > 0 else 0
    return avg_win / abs(avg_loss) if avg_loss != 0 else np.nan


def calculate_net_returns(close_series):
    """
    Calculate net return from a cumulative portfolio value series (assuming start at 1).
    """
    return close_series.iloc[-1] - 1


def calculate_annualized_return(net_return, trading_days):
    """
    Calculate the annualized return given net return and trading days.
    """
    return (1 + net_return) ** (252 / trading_days) - 1


def calculate_annualized_volatility(daily_returns):
    """
    Calculate the annualized volatility from daily returns.
    """
    return daily_returns.std() * np.sqrt(252)


def calculate_sharpe_ratio(daily_returns, risk_free_rate=RISK_FREE_RATE):
    """
    Calculate the Sharpe Ratio.
    """
    excess = daily_returns - risk_free_rate
    return excess.mean() / excess.std() if excess.std() != 0 else np.nan


def calculate_adjusted_sharpe_ratio(daily_returns, risk_free_rate=RISK_FREE_RATE):
    """
    Calculate the Adjusted Sharpe Ratio.
    """
    excess = daily_returns - risk_free_rate
    sharpe = excess.mean() / excess.std() if excess.std() != 0 else np.nan
    skewness = daily_returns.skew()
    kurtosis = daily_returns.kurtosis()
    return sharpe * (1 + (skewness / 6) * sharpe - (kurtosis / 24)) if not np.isnan(sharpe) else np.nan


def calculate_omega_ratio(daily_returns, minimum_return=0):
    """
    Calculate the Omega Ratio.
    """
    gains = daily_returns[daily_returns > minimum_return].sum()
    losses = abs(daily_returns[daily_returns < minimum_return].sum())
    return gains / losses if losses != 0 else np.nan


def calculate_kappa_ratio(daily_returns, n=2, risk_free_rate=RISK_FREE_RATE):
    """
    Calculate the Kappa Ratio.
    """
    excess = daily_returns - risk_free_rate
    downside = (daily_returns[daily_returns < 0] ** n)
    downside_deviation = downside.mean() ** (1/n) if len(downside) > 0 else np.nan
    return excess.mean() / downside_deviation if downside_deviation != 0 else np.nan


def calculate_sortino_ratio(daily_returns, risk_free_rate=RISK_FREE_RATE):
    """
    Calculate the Sortino Ratio.
    """
    excess = daily_returns - risk_free_rate
    downside_std = daily_returns[daily_returns < 0].std()
    return excess.mean() / downside_std if downside_std != 0 else np.nan


def calculate_calmar_ratio(annualized_return, max_drawdown):
    """
    Calculate the Calmar Ratio.
    """
    return annualized_return / abs(max_drawdown) if max_drawdown != 0 else np.nan


# def calculate_treynor_ratio(daily_returns, market_returns, risk_free_rate=RISK_FREE_RATE):
#     """
#     Calculate the Treynor Ratio.
#     """
#     beta = daily_returns.cov(market_returns) / market_returns.var() if market_returns.var() != 0 else np.nan
#     return (daily_returns.mean() - risk_free_rate) / beta if beta not in [0, np.nan] else np.nan


def calculate_treynor_ratio(daily_returns, market_returns, risk_free_rate=0.06/252):
    """
    Calculate the Treynor Ratio.
    
    Changes made:
      - Replaced the 'if beta not in [0, np.nan]' condition with a robust check for beta.
    """
    mkt_var = market_returns.var()
    if mkt_var == 0:
        return np.nan
    beta = daily_returns.cov(market_returns) / mkt_var
    if beta == 0 or np.isnan(beta):
        return np.nan
    return (daily_returns.mean() - risk_free_rate) / beta


def compute_performance_metrics(cum_series, daily_returns, trading_days, market_returns=None, risk_free_rate=0.06/252, ticker=None, best_parameters=None):
    """
    Compute a suite of performance metrics:
      - Net Returns, Annualized Return, Annualized Volatility, Sharpe Ratio,
        Adjusted Sharpe Ratio, Omega Ratio, Kappa Ratio, Sortino Ratio, Calmar Ratio, Treynor Ratio,
        Max Drawdown, Win Rate, Average Trade Return, Win/Loss Ratio.
    """
    net_ret = calculate_net_returns(cum_series)
    ann_ret = calculate_annualized_return(net_ret, trading_days)
    ann_vol = calculate_annualized_volatility(daily_returns)
    sharpe = calculate_sharpe_ratio(daily_returns, risk_free_rate)
    adj_sharpe = calculate_adjusted_sharpe_ratio(daily_returns, risk_free_rate)
    omega = calculate_omega_ratio(daily_returns)
    kappa = calculate_kappa_ratio(daily_returns, n=2, risk_free_rate=risk_free_rate)
    sortino = calculate_sortino_ratio(daily_returns, risk_free_rate)
    max_dd = calculate_max_drawdown(cum_series)
    calmar = calculate_calmar_ratio(ann_ret, max_dd)
    win_rate = calculate_win_rate(daily_returns)
    avg_trade_ret = calculate_avg_trade_return(daily_returns)
    win_loss = calculate_win_loss_ratio(daily_returns)
    
    if market_returns is not None:
        treynor = calculate_treynor_ratio(daily_returns, market_returns, risk_free_rate)
    else:
        treynor = np.nan
        
    return {
        "ticker" : ticker,
        "best_parameters":best_parameters,
        "net_return": net_ret,
        "annualized_return": ann_ret,
        "annualized_volatility": ann_vol,
        "sharpe_ratio": sharpe,
        "adjusted_sharpe_ratio": adj_sharpe,
        "omega_ratio": omega,
        "kappa_ratio": kappa,
        "sortino_ratio": sortino,
        "max_drawdown": max_dd,
        "calmar_ratio": calmar,
        "treynor_ratio": treynor,
        "win_rate": win_rate,
        "avg_trade_return": avg_trade_ret,
        "win_loss_ratio": win_loss
    }