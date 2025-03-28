import pandas as pd
import numpy as np

def max_consecutive_streaks(pnls):
    # Detecting streaks of wins and losses
    streaks = np.sign(pnls)  # 1 for wins, -1 for losses, 0 for no change
    # Cumulative sum of consecutive wins and losses
    win_streaks = (streaks == 1).astype(int).groupby((streaks != streaks.shift()).cumsum()).cumsum()
    loss_streaks = (streaks == -1).astype(int).groupby((streaks != streaks.shift()).cumsum()).cumsum()

    max_win_streak = win_streaks.max()
    max_loss_streak = loss_streaks.max()

    return max_win_streak, max_loss_streak

def calculate_metrics(trades: pd.DataFrame) -> dict:
    # Exclude the last trade
    trades = trades.iloc[:-1]

    # Assuming 'PnL After Commission' is in the trades DataFrame
    if 'PnL After Commission' not in trades.columns:
        raise ValueError("Column 'PnL After Commission' is missing from the trades data.")

    pnls = trades["PnL After Commission"].astype(np.float32)

    total_trades = trades.shape[0]
    wins = pnls[pnls > 0]
    losses = pnls[pnls <= 0]
    net_profit = pnls.sum()
    gross_profit = wins.sum()
    gross_loss = losses.sum()
    average_win = wins.mean() if not wins.empty else 0
    average_loss = losses.mean() if not losses.empty else 0
    average_trade = pnls.mean()

    risk_reward = abs(average_win) / abs(average_loss) if average_loss != 0 else np.nan
    win_rate = (len(wins) / total_trades) * 100 if total_trades != 0 else 0
    profit_factor = abs(gross_profit) / abs(gross_loss) if gross_loss != 0 else np.nan

    sharpe_ratio = pnls.mean() / pnls.std() if pnls.std() != 0 else np.nan
    sortino_ratio = pnls.mean() / np.sqrt(np.mean(np.minimum(0, pnls) ** 2)) if pnls.std() != 0 else np.nan

    max_win_streak, max_loss_streak = max_consecutive_streaks(trades['PnL After Commission'])

    data = {
        "Total Trades": total_trades,
        "Net Profit": net_profit,
        "Gross Profit": gross_profit,
        "Gross Loss": gross_loss,
        "Win Rate": win_rate,
        "Risk Reward": risk_reward,
        "Average Win": average_win,
        "Average Loss": average_loss,
        "Average Trade": average_trade,
        "Sharpe Ratio": sharpe_ratio,
        "Sortino Ratio": sortino_ratio,
        "Profit Factor": profit_factor,
        "Number of Wins": wins.shape[0],
        "Number of Losses": losses.shape[0],
        "Max Consecutive Wins": max_win_streak,
        "Max Consecutive Losses": max_loss_streak
    }

    # Convert np.float32 to Python float for JSON serialization
    data = {k: (float(v) if isinstance(v, np.float32) else v) for k, v in data.items()}
    return data

if __name__ == "__main__":
    trades = pd.read_csv('./tests/executed_trades.csv', index_col="TradeID")
    stats = calculate_metrics(trades)
    for stat, value in stats.items():
         print(f"{stat}: {value}")
