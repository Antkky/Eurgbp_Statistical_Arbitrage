import pandas as pd
import numpy as np
import torch
from collections import deque
from typing import Tuple
from model import LSTM_Q_Net, QTrainer
from environment import TradingEnvironment, Action

MAX_MEMORY = 100_000
BATCH_SIZE = 64
LR = 0.25

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TradingAgent:
  def __init__(self, df: pd.DataFrame, plot: bool = False, debug: bool = False):
    self.debug = debug
    self.epsilon = 0.25
    self.gamma = 0.95
    self.memory = deque(maxlen=MAX_MEMORY)
    self.model = LSTM_Q_Net(input_size=15, hidden_size=128, output_size=4).to(device)
    self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma, batch_size=BATCH_SIZE)
    self.env = TradingEnvironment(df, plot, debug)
    self.equity_curves = {}

  def store_experience(self, state, action, reward, next_state, done):
    self.memory.append((state, action, np.clip(reward, -1, 1), next_state, done))

  def sample_prioritized(self):
    priorities = np.abs([exp[2] for exp in self.memory])
    probs = priorities / np.sum(priorities) if np.sum(priorities) > 0 else np.full(len(priorities), 1 / len(priorities))
    return [self.memory[i] for i in np.random.choice(len(self.memory), BATCH_SIZE, p=probs)]

  def update_epsilon(self):
    self.epsilon = max(0.01, self.epsilon * 0.99)

  def get_action(self, state: pd.Series) -> Tuple[Action, int]:
    state_tensor = torch.tensor(np.nan_to_num(state.values), dtype=torch.float32).unsqueeze(0).to(device)
    action_idx = torch.argmax(self.model(state_tensor)).item()
    return [Action.HOLD, Action.LONG, Action.SHORT, Action.FLATTEN][action_idx], action_idx

  def train(self):
    batch = self.sample_prioritized()
    states, actions, rewards, next_states, dones = zip(*batch)
    self.trainer.train_step(
      torch.tensor(np.array(states), dtype=torch.float32).to(device),
      torch.tensor(actions, dtype=torch.long).to(device),
      torch.tensor(rewards, dtype=torch.float32).to(device),
      torch.tensor(np.array(next_states), dtype=torch.float32).to(device),
      torch.tensor(dones, dtype=torch.bool).to(device)
    )

  def save_model(self, episode):
    self.model.save(f"Episode-{episode}.pth")

  def run(self, episodes: int):
    for episode in range(episodes):
      state, done, total_reward = self.env.reset(), False, 0
      equity_curve = []
      while not done:
        action, action_idx = self.get_action(state)
        state, reward, real_profit, _, done = self.env.step_forward(action)
        equity_curve.append(real_profit)
        self.store_experience(state.values, action_idx, reward, state.values, done)
        self.train()
        total_reward += reward
      self.equity_curves[episode] = equity_curve
      self.update_epsilon()
      self.save_model(episode)

if __name__ == "__main__":
  dates = pd.date_range(start='2020-01-01', periods=100)
  test_data = pd.DataFrame({
    'Asset1_Price': np.random.normal(100, 5, 100),
    'Asset2_Price': np.random.normal(50, 3, 100),
    'Ratio_Price': np.random.normal(2, 0.1, 100),
    'Spread_ZScore': np.random.normal(0, 1, 100),
    'Rolling_Correlation': np.random.uniform(0.5, 0.9, 100),
    'Rolling_Cointegration_Score': np.random.uniform(0.3, 0.7, 100),
    'Asset1_RSI': np.random.uniform(30, 70, 100),
    'Asset2_RSI': np.random.uniform(30, 70, 100),
    'Ratio_RSI': np.random.uniform(30, 70, 100),
    'Asset1_MACD': np.random.normal(0, 1, 100),
    'Asset2_MACD': np.random.normal(0, 1, 100),
    'Ratio_MACD': np.random.normal(0, 1, 100),
    'Hedge_Ratio': np.random.normal(2, 0.05, 100)
  }, index=dates)
  agent = TradingAgent(test_data, plot=True)
  agent.run(10)
