import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
import numpy as np
import pandas as pd
from collections import deque
from typing import List, Tuple, Optional
from enum import Enum

# Constants and Hyperparameters
MAX_MEMORY = 1_000_000
BATCH_SIZE = 1024
LEARNING_RATE = 0.001
GAMMA = 0.99
EPSILON_DECAY = 0.9995
MIN_EPSILON = 0.01
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_float32_matmul_precision('high')

class Action(Enum):
  HOLD = 0
  LONG = 1
  SHORT = 2
  FLATTEN = 3

import torch
import torch.nn as nn
import torch.nn.functional as F

class AdvancedLSTM_Q_Net(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=3,
            batch_first=True,
            dropout=0.2,
            bidirectional=True
        )
        self.layer_norm = nn.LayerNorm(hidden_size * 2)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, output_size)
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)  # Ensure lstm_out is always 3D (batch, seq_len, hidden_size * 2)
        x = self.layer_norm(lstm_out)  # Take last time step's output
        x = self.fc(x)  #Pass through fully connected layers
        return x.squeeze()  # Ensure output is 1D


class PrioritizedReplayBuffer:
  def __init__(self, capacity: int, alpha: float = 0.6, beta: float = 0.4):
    self.memory = deque(maxlen=capacity)
    self.priorities = deque(maxlen=capacity)
    self.alpha = alpha
    self.beta = beta

  def add(self, experience, priority=1.0):
    self.memory.append(experience)
    self.priorities.append(priority)

  def sample(self, batch_size: int):
    priorities = np.array(self.priorities, dtype=np.float32) ** self.alpha
    probs = priorities / priorities.sum()
    indices = np.random.choice(len(self.memory), batch_size, p=probs, replace=False)
    samples = [self.memory[idx] for idx in indices]
    weights = (len(self.memory) * probs[indices]) ** -self.beta
    weights /= weights.max()
    return samples, indices, torch.tensor(weights, dtype=torch.float32, device=DEVICE)

class AdvancedTradingAgent:
  def __init__(self, input_size: int, output_size: int, lr: float = LEARNING_RATE):
    self.model = AdvancedLSTM_Q_Net(input_size, 128, output_size).to(DEVICE)
    self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-5, amsgrad=True)
    self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=10, T_mult=2)
    self.scaler = GradScaler()
    self.replay_buffer = PrioritizedReplayBuffer(MAX_MEMORY)
    self.epsilon = 0.25
    self.step_count = 0

  def select_action(self, state: torch.Tensor) -> int:
    self.step_count += 1
    self.epsilon = max(MIN_EPSILON, self.epsilon * EPSILON_DECAY)
    if np.random.random() < self.epsilon:
      return np.random.randint(len(Action))
    with torch.no_grad():
      return self.model(state.to(DEVICE)).argmax().item()

  def train(self, batch_size: int = BATCH_SIZE):
    if len(self.replay_buffer.memory) < batch_size:
      return None
    samples, _, weights = self.replay_buffer.sample(batch_size)
    states, actions, rewards, next_states, dones = zip(*samples)

    states = torch.stack(states).to(DEVICE)
    next_states = torch.stack(next_states).to(DEVICE)
    actions = torch.tensor(actions, dtype=torch.long, device=DEVICE)
    rewards = torch.tensor(rewards, dtype=torch.float32, device=DEVICE)
    dones = torch.tensor(dones, dtype=torch.bool, device=DEVICE)

    with autocast():
      current_q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
      next_q_values = self.model(next_states).max(1)[0]
      expected_q_values = rewards + (1 - dones.float()) * GAMMA * next_q_values
      loss = F.smooth_l1_loss(current_q_values, expected_q_values, reduction='none')
      loss = (loss * weights).mean()

    self.optimizer.zero_grad()
    self.scaler.scale(loss).backward()
    self.scaler.step(self.optimizer)
    self.scaler.update()
    self.scheduler.step()
    return loss.item()

  def save_model(self, filepath: str):
    torch.save(self.model.state_dict(), filepath)

  def load_model(self, filepath: str):
    self.model.load_state_dict(torch.load(filepath))
    self.model.eval()

def load_and_preprocess_data(file_path: str) -> Optional[torch.Tensor]:
  try:
    df = pd.read_csv(file_path)
    numeric_data = df.select_dtypes(include=[np.number])
    return torch.tensor(numeric_data.values, dtype=torch.float32, device=DEVICE)
  except Exception as e:
    print(f"Data loading error: {e}")
    return None

def train_agent(agent: AdvancedTradingAgent, data: torch.Tensor, episodes: int = 50):
  for episode in range(episodes):
    state = data[0]
    total_reward = 0
    for step in range(len(data)):
      action = agent.select_action(state)
      next_state = data[min(step + 1, len(data) - 1)]
      reward = 0  # Define your reward logic
      done = step == len(data) - 1
      agent.replay_buffer.add((state, action, reward, next_state, done))
      loss = agent.train()
      state = next_state
      total_reward += reward
    print(f"Episode {episode}: Total Reward = {total_reward}")
    if episode % 10 == 0:
      agent.save_model(f'trading_model_episode_{episode}.pth')

def main():
  data_path = 'data/processed/train_data_scaled.csv'
  data = load_and_preprocess_data(data_path)
  if data is None:
    print("Failed to load data. Exiting.")
    return
  agent = AdvancedTradingAgent(input_size=data.shape[1], output_size=len(Action))
  train_agent(agent, data)

if __name__ == "__main__":
  main()
