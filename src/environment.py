import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from enum import Enum
from typing import Tuple, List, Optional

class Action(Enum):
  HOLD = 0
  LONG = 1
  SHORT = 2
  FLATTEN = 3

class TradeSide(Enum):
  LONG = 0
  SHORT = 1

class Position:
  def __init__(self, entry_price: float, size: int, side: TradeSide):
    self.entry_price = entry_price
    self.size = size
    self.side = side

  def get_unrealized_pnl(self, current_price: float) -> float:
    diff = (current_price - self.entry_price) if self.side == TradeSide.LONG else (self.entry_price - current_price)
    return self.entry_price * self.size * (diff / ((self.entry_price + current_price) / 2) / 100)

  def close(self, current_price: float) -> float:
    return self.get_unrealized_pnl(current_price)

class TradingEnvironment:
  REQUIRED_FEATURES = [
    "Asset1_Price", "Asset2_Price", "Ratio_Price", "Spread_ZScore", "Rolling_Correlation",
    "Rolling_Cointegration_Score", "Asset1_RSI", "Asset2_RSI", "Ratio_RSI",
    "Asset1_MACD", "Asset2_MACD", "Ratio_MACD"
  ]

  PERFORMANCE_COLUMNS = ["Unrealized_PnL", "Realized_PnL", "Positioned"]

  def __init__(self, df: pd.DataFrame, plot: bool = False, debug: bool = False):
    self.step = 0
    self.window_size = 1
    self.debug = debug
    self.plot = plot
    self.balance = self.equity = 1_000_000
    self.positions: List[Tuple[Position, Position]] = []
    self.realized_pnl = self.unrealized_pnl = 0
    self.trade_history, self.reward_history = [], []
    self.buy_signals, self.sell_signals = [], []
    self.last_trade_step = 0
    self._prepare_data(df)

  def _prepare_data(self, df: pd.DataFrame):
    missing_features = [f for f in self.REQUIRED_FEATURES if f not in df.columns]
    if missing_features:
      raise ValueError(f"Missing features: {missing_features}")
    self.data = df.copy()[self.REQUIRED_FEATURES]
    for col in self.PERFORMANCE_COLUMNS:
      self.data[col] = 0
    if "Hedge_Ratio" not in df.columns:
      raise ValueError("Missing 'Hedge_Ratio' column")
    self.hedge_ratio = df["Hedge_Ratio"]

  def reset(self) -> pd.Series:
    self.step = self.realized_pnl = self.unrealized_pnl = 0
    self.balance = self.equity = 1_000_000
    self.positions.clear()
    self.trade_history.clear()
    self.reward_history.clear()
    self.buy_signals.clear()
    self.sell_signals.clear()
    self.last_trade_step = 0
    return self.current_observation()

  def step_forward(self, action: Action) -> Tuple[pd.Series, float, float, float, bool]:
    self.step += 1
    self.last_trade_step += 1
    reward, profit, positioned = self.execute_trade(action)
    self.trade_history.append(self.realized_pnl)
    self.reward_history.append(reward)
    done = self.step + self.window_size >= len(self.data)
    obs = self.current_observation()
    obs.update({"Unrealized_PnL": self.unrealized_pnl, "Realized_PnL": self.realized_pnl, "Positioned": positioned})
    return obs, reward, self.realized_pnl, self.unrealized_pnl, done

  def execute_trade(self, action: Action) -> Tuple[float, float, int]:
    asset1_price = self.data.iloc[self.step]["Asset1_Price"]
    asset2_price = self.data.iloc[self.step]["Asset2_Price"]
    hedge_ratio = self.hedge_ratio.iloc[self.step]
    profit = reward = 0
    if action == Action.HOLD:
      self.update_unrealized_pnl(asset1_price, asset2_price)
    elif action in (Action.LONG, Action.SHORT) and not self.positions:
      size = 100_000
      positions = (
        Position(asset1_price, size, TradeSide.SHORT if action == Action.LONG else TradeSide.LONG),
        Position(asset2_price, int(size * hedge_ratio), TradeSide.LONG if action == Action.LONG else TradeSide.SHORT)
      )
      self.positions.append(positions)
      (self.buy_signals if action == Action.LONG else self.sell_signals).append(self.step)
      self.last_trade_step = 0
    elif action == Action.FLATTEN and self.positions:
      pnl = sum(pos_a.close(asset1_price) + pos_b.close(asset2_price) for pos_a, pos_b in self.positions)
      self.realized_pnl += pnl
      self.unrealized_pnl = 0
      self.positions.clear()
      profit = reward = pnl
    return reward, profit, int(bool(self.positions))

  def update_unrealized_pnl(self, asset1_price: float, asset2_price: float):
    self.unrealized_pnl = sum(
      pos_a.get_unrealized_pnl(asset1_price) + pos_b.get_unrealized_pnl(asset2_price)
      for pos_a, pos_b in self.positions
    )

  def current_observation(self) -> pd.Series:
    return self.data.iloc[self.step]
