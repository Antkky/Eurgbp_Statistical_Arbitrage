import backtrader as bt
import numpy as np
import csv
import os
import datetime
from collections import deque
from statsmodels.tsa.stattools import coint

class ARS(bt.Strategy):
    def __init__(self):
        # Trade logging setup
        self.version = "0.0.1"


        self.csv_file = 'trade_log.csv'
        self.csv_headers = ['TradeID', 'OpenDateTime', 'CloseDateTime', 'Size',
                            'Commission', 'EntryPrice', 'ExitPrice', 'PnL', 'PnLAfterCommission']

        # Initialize CSV file with headers if it doesn't exist
        if not os.path.exists(self.csv_file) or os.stat(self.csv_file).st_size == 0:
            with open(self.csv_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(self.csv_headers)

        # Track open trades with their details
        self.open_trades = {}
        self.trade_id_counter = 1

        # Strategy components
        self.spread = deque(maxlen=60)
        self.correlation = deque(maxlen=15)
        self.cointegration = deque(maxlen=60)

        self.EURUSDh = deque(maxlen=60)
        self.GBPUSDh = deque(maxlen=60)

        self.EURUSD = self.datas[0]
        self.GBPUSD = self.datas[1]

    def next(self):
        self.EURUSDh.append(self.EURUSD.close[0])
        self.GBPUSDh.append(self.GBPUSD.close[0])

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
        try:
            if np.std(eurusd_list) > 0 and np.std(gbpusd_list) > 0:
                coint_t, coint_p_values, critical_values = coint(eurusd_list, gbpusd_list)
                return coint_p_values
        except Exception as e:
            print(f"Cointegration error: {e}")
        return 1.0

    def calculate_correlation(self, eurusd_list, gbpusd_list):
        try:
            correlation_coefficient = np.corrcoef(eurusd_list[-15:], gbpusd_list[-15:])[0, 1]
            return 0 if np.isnan(correlation_coefficient) else correlation_coefficient
        except Exception as e:
            print(f"Correlation error: {e}")
            return 0

    def exit_logic(self, spread_zscore):
        if abs(spread_zscore) < 0.1 and self.position != 0:
            self.close()

    def entry_logic(self, coint_p_values, correlation_coefficient, spread_zscore):
      try:
        lotsize = 1.5
        if coint_p_values < 0.05 and min(self.correlation) < 0.7:
          if spread_zscore < -2:
            self.buy(size=lotsize * 100000, exectype=bt.Order.Market)
          elif spread_zscore > 2:
            self.sell(size=lotsize * 100000, exectype=bt.Order.Market)
      except Exception as e:
        print(f"Error making trade entry: {e}")

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
