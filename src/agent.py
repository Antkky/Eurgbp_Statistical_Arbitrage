import pandas as pd
import numpy as np
import torch
from collections import deque
import os
from tqdm import tqdm
from typing import Tuple
from src.model import LSTM_Q_Net, QTrainer
from src.environment import TradingEnvironment, Action

# Hyperparameters
MAX_MEMORY = 100_000
BATCH_SIZE = 128
LEARNING_RATE = 0.25
GAMMA = 0.95
EPSILON_DECAY = 0.99
MIN_EPSILON = 0.01

# Device Configuration with faster CUDA operations
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == 'cuda':
    torch.backends.cudnn.benchmark = True # Optimize CUDA operations

class TradingAgent:
    def __init__(self, data: pd.DataFrame, plot: bool = False, debug: bool = False):
        self.debug = debug
        self.epsilon = 0.25
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = LSTM_Q_Net(input_size=15, hidden_size=128, output_size=4).to(device)
        self.trainer = QTrainer(self.model, lr=LEARNING_RATE, gamma=GAMMA, batch_size=BATCH_SIZE)
        self.env = TradingEnvironment(data, plot, debug)
        self.length = len(data)
        self.equity_curves = {}

    def store_experience(self, state, action, reward, next_state, done):
      if reward is None or np.isnan(reward):  # Handle None or NaN rewards
          print("DEBUG: Received None or NaN reward, setting to 0.0")
          reward = 0.0
      else:
          reward = np.clip(reward, -1, 1)

      self.memory.append((state, action, reward, next_state, done))


    def sample_experiences(self):
        if len(self.memory) < BATCH_SIZE:
            return list(self.memory)  # Return whatever is available

        priorities = np.abs(np.array([exp[2] for exp in self.memory]))  # Absolute rewards
        sum_priorities = np.sum(priorities)

        # Ensure priorities are non-zero by adding a small constant
        if sum_priorities == 0:
            probabilities = np.ones(len(self.memory)) / len(self.memory)  # Uniform sampling
        else:
            probabilities = (priorities + 1e-6) / (sum_priorities + 1e-6)  # Avoid division by zero

        # FIX: Normalize probabilities to ensure they sum to exactly 1
        probabilities = probabilities / np.sum(probabilities)

        sample_size = min(BATCH_SIZE, len(self.memory))
        indices = np.random.choice(len(self.memory), sample_size, p=probabilities, replace=False)

        return [self.memory[i] for i in indices]

    def update_epsilon(self):
        self.epsilon = max(MIN_EPSILON, self.epsilon * EPSILON_DECAY)

    @torch.no_grad() # Disable gradient tracking for inference
    def select_action(self, state: pd.Series) -> Tuple[Action, int]:
        # Epsilon-greedy action selection
        if np.random.random() < self.epsilon:
            action_idx = np.random.randint(0, 4)
        else:
            # Single conversion to tensor with proper handling of NaN values
            state_tensor = torch.tensor(np.nan_to_num(state.values),
                                      dtype=torch.float32).unsqueeze(0).to(device)
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
        )  # Fixed: Added missing closing parenthesis

    def save_model(self, episode: int):
      """
      Saves the trained model with proper error handling.
      """
      try:
          if episode is not None:
              print(f"DEBUG: Saving model for episode {episode}")
              self.model.save(f"Episode-{episode}.pth")
          else:
              print("DEBUG: Episode variable is None, using fallback name")
              self.model.save("Episode-unknown.pth")
      except Exception as e:
          print(f"DEBUG: Error in save_model - {str(e)}")

    def run(self, episodes: int):
        progress = tqdm(total=self.length)
        for episode in range(episodes):
            state, done = self.env.reset(), False
            equity_curve = []
            while not done:
                progress.update(1)
                action, action_idx = self.select_action(state)
                next_state, reward, real_profit, _, done = self.env.step_forward(action)
                equity_curve.append(real_profit)

                # Store experience using NumPy arrays directly
                self.store_experience(state.values, action_idx, reward, next_state.values, done)
                self.train()
                state = next_state

            self.equity_curves[episode] = equity_curve
            self.update_epsilon()
            self.save_model(episode)  # Now properly handles None case
