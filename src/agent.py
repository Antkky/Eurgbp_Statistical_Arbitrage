import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from collections import deque
from typing import Tuple
from src.model import LSTM_Q_Net, QTrainer
from src.environment import TradingEnvironment, Action

# Hyperparameters
MAX_MEMORY = 150_000
BATCH_SIZE = 128
LEARNING_RATE = 0.5
GAMMA = 0.95
EPSILON_DECAY = 0.98
MIN_EPSILON = 0.01

# Device Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TradingAgent:
  """
  Deep Q-Learning based trading agent using LSTM.
  """
  def __init__(self, data: pd.DataFrame, plot: bool = False, debug: bool = False):
    """
    Initializes the trading agent with model, trainer, and environment.

    :param data: Market data as a Pandas DataFrame
    :param plot: Whether to enable plotting
    :param debug: Whether to enable debugging mode
    """
    self.debug = debug
    self.epsilon = 0.25
    self.memory = deque(maxlen=MAX_MEMORY)
    self.model = LSTM_Q_Net(input_size=15, hidden_size=128, output_size=4).to(device)
    self.trainer = QTrainer(self.model, lr=LEARNING_RATE, gamma=GAMMA, batch_size=BATCH_SIZE)
    self.env = TradingEnvironment(data, plot, debug)
    self.datalength = len(data)
    self.equity_curves = {}

  def store_experience(self, state, action, reward, next_state, done):
    """
    Stores experience in memory with clipped reward.
    """
    self.memory.append((state, action, np.clip(reward, -1, 1), next_state, done))

  def sample_experiences(self):
    """
    Samples a mini-batch of experiences using prioritized experience replay.
    """
    priorities = np.abs([exp[2] for exp in self.memory])
    probabilities = priorities / np.sum(priorities) if np.sum(priorities) > 0 else np.full(len(priorities), 1 / len(priorities))
    return [self.memory[i] for i in np.random.choice(len(self.memory), BATCH_SIZE, p=probabilities)]

  def update_epsilon(self):
    """
    Decays epsilon to reduce exploration over time.
    """
    self.epsilon = max(MIN_EPSILON, self.epsilon * EPSILON_DECAY)

  def select_action(self, state: pd.Series) -> Tuple[Action, int]:
    """
    Selects an action using the trained model.

    :param state: Current state as a Pandas Series
    :return: Tuple containing the action and its index
    """
    state_tensor = torch.tensor(np.nan_to_num(state.values), dtype=torch.float32).unsqueeze(0).to(device)
    action_idx = torch.argmax(self.model(state_tensor)).item()
    return [Action.HOLD, Action.LONG, Action.SHORT, Action.FLATTEN][action_idx], action_idx

  def train(self):
    """
    Trains the model using a mini-batch from experience replay.
    """
    batch = self.sample_experiences()
    states, actions, rewards, next_states, dones = zip(*batch)
    self.trainer.train_step(
      torch.tensor(np.array(states), dtype=torch.float32).to(device),
      torch.tensor(actions, dtype=torch.long).to(device),
      torch.tensor(rewards, dtype=torch.float32).to(device),
      torch.tensor(np.array(next_states), dtype=torch.float32).to(device),
      torch.tensor(dones, dtype=torch.bool).to(device)
    )

  def save_model(self, episode: int):
    """
    Saves the trained model.
    """
    self.model.save(f"Episode-{episode}.pth")

  def run(self, episodes: int):
    """
    Executes training over multiple episodes.

    :param episodes: Number of episodes to train the agent
    """
    for episode in tqdm(range(episodes), desc="Training Progress"):
        state, done, total_reward = self.env.reset(), False, 0
        equity_curve = []
        for step in tqdm(range(self.datalength), desc=f"Episode {episode+1}", leave=False):
            action, action_idx = self.select_action(state)
            state, reward, real_profit, _, done = self.env.step_forward(action)
            equity_curve.append(real_profit)
            self.store_experience(state.values, action_idx, reward, state.values, done)
            self.train()
            total_reward += reward
        self.equity_curves[episode] = equity_curve
        self.update_epsilon()
        self.save_model(episode)

if __name__ == "__main__":
  # Generate test data
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
