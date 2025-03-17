import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LSTM_Q_Net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout=0.3, bidirectional=False):
        super(LSTM_Q_Net, self).__init__()
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional

        # Fixed: Added missing closing parenthesis
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=2, # Reduced from 3 for faster processing
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional,
        )  # Added closing parenthesis here

        mult = 2 if bidirectional else 1
        self.ln = nn.LayerNorm(hidden_size * mult)
        self.fc1 = nn.Linear(hidden_size * mult, 128)
        self.fc2 = nn.Linear(128, output_size)

    def forward(self, x):
        # Handle both single samples and batches efficiently
        if len(x.shape) == 2:
            x = x.unsqueeze(1) # Add sequence dimension if missing

        lstm_out, _ = self.lstm(x)

        # Use the last timestep output
        last_step = lstm_out[:, -1, :]
        normalized = self.ln(last_step)

        # Fully connected layers
        x = F.relu(self.fc1(normalized))
        return self.fc2(x)

    def save(self, file_name):
        model_folder_path = "./models"
        os.makedirs(model_folder_path, exist_ok=True)
        file_path = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_path)

class QTrainer:
    def __init__(self, model, lr, gamma, batch_size, target_update_freq=50):
        self.lr = lr
        self.gamma = gamma
        self.model = model.to(device)

        # Create target network with same architecture
        self.target_model = LSTM_Q_Net(
            input_size=15,
            hidden_size=model.hidden_size,
            output_size=model.fc2.out_features,
            bidirectional=model.bidirectional
        ).to(device)

        self.update_target()

        # Use Adam with improved parameters
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr, amsgrad=True)

        # Less frequent LR adjustments
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=20, gamma=0.5)

        # Huber loss (SmoothL1Loss) is more robust for RL
        self.criterion = nn.SmoothL1Loss()

        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.train_step_count = 0

    def update_target(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def train_step(self, state, action, reward, next_state, done):
        # Reshape action for gathering
        action = action.view(-1, 1)

        # Get current Q values
        pred = self.model(state)

        # Implement Double Q-learning for stability
        with torch.no_grad():
            # Get actions from main network
            next_actions = self.model(next_state).argmax(dim=1, keepdim=True)

            # Get Q-values from target network
            next_q_values = self.target_model(next_state).gather(1, next_actions)

            # Compute target Q values
            target = reward.unsqueeze(1) + (1 - done.float().unsqueeze(1)) * self.gamma * next_q_values

        # Get Q values for taken actions
        q_values = pred.gather(1, action)

        # Calculate loss and optimize
        loss = self.criterion(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()

        # Apply gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        self.optimizer.step()

        # Update target network less frequently
        self.train_step_count += 1
        if self.train_step_count % self.target_update_freq == 0:
            self.update_target()
            self.scheduler.step()
