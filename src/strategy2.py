import backtrader as bt
import numpy as np
import csv
from collections import deque
from statsmodels.tsa.stattools import coint

class ARS(bt.Strategy):
  def __init__(self):
    self.trade_log = []

    self.spread = deque(maxlen=60)
    self.correlation = deque(maxlen=15)
    self.cointegration = deque(maxlen=60)

    self.EURUSDh = deque(maxlen=60)
    self.GBPUSDh = deque(maxlen=60)

    self.EURUSD = self.getdatabyname("EURUSD")
    self.GBPUSD = self.getdatabyname("GBPUSD")

  def next(self):
    self.EURUSDh.append(self.EURUSD[0])
    self.GBPUSDh.append(self.GBPUSD[0])
    if len(self.EURUSDh) < 60 or len(self.GBPUSDh) < 60:
      return

    eurusd_list = list(self.EURUSDh)
    gbpusd_list = list(self.GBPUSDh)

    coint_p_values = self.calculate_cointegration(eurusd_list, gbpusd_list)
    self.cointegration.append(coint_p_values)

    correlation_coefficient = self.calculate_correlation(eurusd_list, gbpusd_list)
    self.correlation.append(correlation_coefficient)

    self.spread.append(eurusd_list[-1] - gbpusd_list[-1])
    spread_zscore = self.spread_zscore(self.spread)

    self.exit_logic(spread_zscore)
    self.entry_logic(coint_p_values, correlation_coefficient, spread_zscore)

  def calculate_cointegration(self, eurusd_list, gbpusd_list):
    if np.std(eurusd_list) > 0 and np.std(gbpusd_list) > 0 and not np.allclose(eurusd_list, eurusd_list[0]) and not np.allclose(gbpusd_list, gbpusd_list[0]):
      try:
        coint_t, coint_p_values, critical_values = coint(eurusd_list, gbpusd_list)
        return coint_p_values
      except Exception as e:
        print(f"Cointegration error: {e}")
    else:
      print("Skipping cointegration test: Series is constant or nearly identical")
    return 1.0

  def calculate_correlation(self, eurusd_list, gbpusd_list):
    try:
      correlation_coefficient = np.corrcoef(eurusd_list[-15:], gbpusd_list[-15:])[0, 1]
      return 0 if np.isnan(correlation_coefficient) else correlation_coefficient
    except Exception as e:
      print(f"Correlation error: {e}")
      return 0

  def exit_logic(self, spread_zscore):
    if abs(spread_zscore) < 0.1 and self.position:
      self.close(self.getdatabyname("Ratio"))

  def entry_logic(self, coint_p_values, correlation_coefficient, spread_zscore):
    try:
      lotsize = 1.5
      if coint_p_values < 0.05 and any(x < 0.7 for x in self.correlation):
        if spread_zscore < -2:
          self.buy(data=self.getdatabyname("Ratio"), size=lotsize * 100000)
        elif spread_zscore > 2:
          self.sell(data=self.getdatabyname("Ratio"), size=lotsize * 100000)
    except Exception as e:
      print(f"Error making trade entry: {e}")

  def notify_trade(self, trade):
    if trade.justopened:
      self.trade_log.append({
        "tradeID": trade.ref,
        "open_datetime": self.data.datetime.datetime(0),
        "close_datetime": None,
        "size": trade.size,
        "commission": trade.commission,
        "entry_price": trade.price,
        "exit_price": None,
        "pnl": None,
        "pnl_after_commission": None,
      })
    if trade.isclosed:
      for trd in self.trade_log:
        if trd["tradeID"] == trade.ref:
          trd.update({
            "close_datetime": self.data.datetime.datetime(0),
            "exit_price": trade.price,
            "pnl": trade.pnl,
            "pnl_after_commission": trade.pnlcomm,
          })

  def stop(self):
    with open("./tests/latest/trades.csv", "w", newline="") as f:
      writer = csv.writer(f)
      writer.writerow([
        "TradeID", "Open Datetime", "Close Datetime", "Size", "Commission",
        "Entry Price", "Exit Price", "PnL", "PnL After Commission"
      ])
      for trade in self.trade_log:
        writer.writerow([
          trade["tradeID"], trade["open_datetime"], trade["close_datetime"],
          trade["size"], trade["commission"], trade["entry_price"],
          trade["exit_price"], trade["pnl"], trade["pnl_after_commission"]
        ])

  @staticmethod
  def spread_zscore(spreads) -> float:
    try:
      if len(spreads) > 1:
        spread_mean = np.mean(spreads)
        spread_std = np.std(spreads)
        if spread_std > 1e-6:
          return (spreads[-1] - spread_mean) / spread_std
      return 0
    except Exception as e:
      print(f"Spread Z-Score calculation error: {e}")
      return 0
