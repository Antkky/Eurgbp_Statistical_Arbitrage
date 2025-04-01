# take in a dataset, iterate through the data, save & fetch positions efficiently
# Optimized for maximum performance using numba and numpy

import numpy as np
from enum import IntEnum
from numba import jit, float64, int64, boolean, types
from numba.experimental import jitclass
from numba.typed import List, Dict

class Actions(IntEnum):
    HOLD = 0
    LONG = 1
    SHORT = 2
    FLATTEN = 3

position_spec = [
    ('entry_price', float64),
    ('size', float64),
    ('position_type', int64),
    ('entry_step', int64),
    ('exit_price', float64),
    ('exit_step', int64),
    ('is_open', boolean),
    ('pnl', float64)
]

@jitclass(position_spec)
class Position:
    def __init__(self, entry_price, size, position_type, entry_step):
        self.entry_price = entry_price
        self.size = size
        self.position_type = position_type  # 1 for LONG, 2 for SHORT
        self.entry_step = entry_step
        self.exit_price = 0.0
        self.exit_step = -1
        self.is_open = True
        self.pnl = 0.0
    
    def get_unrealized(self, current_price):
        if not self.is_open:
            return 0.0
        
        if self.position_type == 1:
            return self.size * (current_price - self.entry_price)
        else:
            return self.size * (self.entry_price - current_price)
        
    def close(self, exit_price, exit_step):
        self.exit_price = exit_price
        self.exit_step = exit_step
        self.is_open = False
        
        if self.position_type == 1:
            self.pnl = self.size * (exit_price - self.entry_price)
        else:
            self.pnl = self.size * (self.entry_price - exit_price)
        
        return self.pnl

class Environment:
    def __init__(self, df, window_size, position_size=1.0):
        self.data = df.values.astype(np.float64)
        self.window_size = window_size
        self.total_steps = len(df)
        self.step = window_size - 1

        self.asset1_col = df.columns.get_loc("asset1_price")
        self.asset2_col = df.columns.get_loc("asset2_price")
        
        self.position_size = position_size
        self.trade_streaks = np.zeros(self.total_steps, dtype=np.int64)
        self.unrealized = np.zeros(self.total_steps, dtype=np.float64)
        self.realized = np.zeros(self.total_steps, dtype=np.float64)
        self.equity_curve = np.zeros(self.total_steps, dtype=np.float64)
        self.equity_curve[0] = 1000.0
        self.action_history = np.zeros(self.total_steps, dtype=np.int8)
        
        self.current_position_type = 0
        self.position_entry_price = 0.0
        self.position_entry_step = -1
        self.current_streak = 0
        self.total_pnl = 0.0
        self.max_drawdown = 0.0
        
        self.num_trades = 0
        self.winning_trades = 0
        self.total_trade_pnl = 0.0
        
        self.state_buffer = np.zeros((window_size, df.shape[1]), dtype=np.float64)
        
        self._init_jit_functions()
    
    def _init_jit_functions(self):
        test_arr = np.zeros((5, 5), dtype=np.float64)
        _ = self._get_window_fast(test_arr, 0, 5)
        _ = self._calculate_sharpe_fast(np.array([0.01, 0.02, -0.01]), 0.0)
    
    def reset(self):
        self.step = self.window_size - 1
        self.current_position_type = 0
        self.position_entry_price = 0.0
        self.position_entry_step = -1
        self.current_streak = 0
        self.trade_streaks.fill(0)
        self.unrealized.fill(0)
        self.realized.fill(0)
        self.equity_curve.fill(0)
        self.equity_curve[0] = 1000.0
        self.action_history.fill(0)
        self.total_pnl = 0.0
        self.max_drawdown = 0.0
        self.num_trades = 0
        self.winning_trades = 0
        self.total_trade_pnl = 0.0
        
        return self._getstate()
    
    @staticmethod
    @jit(nopython=True)
    def _get_window_fast(data, start_idx, end_idx):
        return data[max(0, start_idx):end_idx].copy()
    
    @staticmethod
    @jit(nopython=True)
    def _calculate_sharpe_fast(returns, risk_free_rate):
        if len(returns) < 2:
            return 0.0
        
        std = np.std(returns)
        if std == 0:
            return 0.0
            
        return (np.mean(returns) - risk_free_rate) / std * np.sqrt(252)
    
    @jit(nopython=True)
    def _get_price(self, step, col_idx):
        if 0 <= step < self.data.shape[0] and 0 <= col_idx < self.data.shape[1]:
            return self.data[step, col_idx]
        return 0.0
    
    def forward(self, action):
        self.action_history[self.step] = action
        self._execute_fast(action)
        self._update_fast()
        reward = self._calculate_reward_fast()
        self.step += 1
        done = self.step >= self.total_steps - 1
        
        if not done:
            new_state = self._getstate()
        else:
            if self.current_position_type != 0:
                self._execute_fast(Actions.FLATTEN)
            new_state = None
        
        return new_state, reward, done
    
    def _execute_fast(self, action):
        asset1_price = self.data[self.step, self.asset1_col]
        asset2_price = self.data[self.step, self.asset2_col]
        
        if action == Actions.FLATTEN and self.current_position_type != 0:
            if self.current_position_type == 1:
                exit_price = asset1_price
            else:
                exit_price = asset2_price
                
            if self.current_position_type == 1:
                pnl = self.position_size * (exit_price - self.position_entry_price)
            else:
                pnl = self.position_size * (self.position_entry_price - exit_price)
            
            self.realized[self.step] = pnl
            self.total_pnl += pnl
            self.num_trades += 1
            self.total_trade_pnl += pnl
            
            if pnl > 0:
                self.winning_trades += 1
                self.current_streak = max(1, self.current_streak + 1)
            else:
                self.current_streak = min(-1, self.current_streak - 1)
            self.current_position_type = 0
            
        elif action == Actions.LONG and self.current_position_type == 0:
            self.current_position_type = 1
            self.position_entry_price = asset1_price
            self.position_entry_step = self.step
            self.current_streak = 1 if self.current_streak >= 0 else -1
            
        elif action == Actions.SHORT and self.current_position_type == 0:
            self.current_position_type = 2
            self.position_entry_price = asset2_price
            self.position_entry_step = self.step
            self.current_streak = 1 if self.current_streak >= 0 else -1
    
    def _update_fast(self):
        self.trade_streaks[self.step] = self.current_streak
        if self.current_position_type != 0:
            if self.current_position_type == 1:
                current_price = self.data[self.step, self.asset1_col]
                unrealized_pnl = self.position_size * (current_price - self.position_entry_price)
            else:
                current_price = self.data[self.step, self.asset2_col]
                unrealized_pnl = self.position_size * (self.position_entry_price - current_price)
                
            self.unrealized[self.step] = unrealized_pnl
        else:
            self.unrealized[self.step] = 0.0
        
        if self.step > 0:
            self.equity_curve[self.step] = (
                self.equity_curve[self.step-1] + 
                self.realized[self.step] + 
                self.unrealized[self.step] - 
                self.unrealized[self.step-1]
            )
            
            current_equity = self.equity_curve[self.step]
            previous_peak = np.max(self.equity_curve[:self.step+1])
            
            current_drawdown = (previous_peak - current_equity) / previous_peak if previous_peak > 0 else 0
            self.max_drawdown = max(self.max_drawdown, current_drawdown)
    
    def _calculate_reward_fast(self):
        reward = self.unrealized[self.step] + self.realized[self.step]
        
        if self.current_position_type != 0:
            holding_time = self.step - self.position_entry_step
            if holding_time > 20:
                reward -= 0.01 * holding_time
        
        return reward
    
    def _getstate(self):
        start_idx = max(0, self.step - self.window_size + 1)
        end_idx = self.step + 1
        window = self._get_window_fast(self.data, start_idx, end_idx)
        state = np.zeros(6, dtype=np.float64)
        state[0] = self.current_position_type
        
        if self.current_position_type != 0:
            state[1] = self.step - self.position_entry_step
            state[2] = self.unrealized[self.step]
        else:
            state[1] = 0
            state[2] = 0
            
        state[3] = self.current_streak
        state[4] = self.realized[self.step]
        
        if self.step > 0:
            current_equity = self.equity_curve[self.step]
            previous_peak = np.max(self.equity_curve[:self.step+1])
            state[5] = (previous_peak - current_equity) / previous_peak if previous_peak > 0 else 0
        else:
            state[5] = 0
        
        return {'window': window, 'features': state}
    
    def get_performance_metrics(self):
        returns = np.diff(self.equity_curve[:self.step+1]) / np.maximum(1e-8, self.equity_curve[:self.step])
        win_rate = self.winning_trades / max(1, self.num_trades)
        avg_profit = self.total_trade_pnl / max(1, self.num_trades)
        sharpe = self._calculate_sharpe_fast(returns, 0.0)
        
        return {
            'total_pnl': self.total_pnl,
            'max_drawdown': self.max_drawdown,
            'sharpe_ratio': sharpe,
            'win_rate': win_rate,
            'avg_profit_per_trade': avg_profit,
            'num_trades': self.num_trades
        }