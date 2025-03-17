import torch # type: ignore
import torch.nn as nn # type: ignore
import torch.optim as optim # type: ignore
import torch.nn.functional as F # type: ignore
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LSTM_Q_Net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout=0.5, bidirectional=True):
        super(LSTM_Q_Net, self).__init__()
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=3,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional,
        )

        self.ln = nn.LayerNorm(hidden_size * 2 if bidirectional else hidden_size)
        self.fc1 = nn.Linear(hidden_size * 2 if bidirectional else hidden_size, 128)
        self.fc2 = nn.Linear(128, output_size)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
      lstm_out, _ = self.lstm(x)

      # If input is (batch_size, hidden_size), don't index for seq_len
      if len(lstm_out.shape) == 3:
        lstm_out = lstm_out[:, -1, :]  # Extract last timestep (only works for sequences)
      else:
        lstm_out = lstm_out  # No need to slice when seq_len is missing

      lstm_out = self.ln(lstm_out)
      x = F.relu(self.fc1(lstm_out))
      x = self.fc2(x)
      return x


    def save(self, file_name):
        model_folder_path = "./models"
        os.makedirs(model_folder_path, exist_ok=True)
        file_path = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_path)


class QTrainer:
  def __init__(self, model, lr, gamma, batch_size, target_update_freq=10):
    self.lr = lr
    self.gamma = gamma
    self.model = model.to(device)

    # Ensure target model matches the main model exactly
    self.target_model = LSTM_Q_Net(
      input_size=15,
      hidden_size=model.hidden_size,
      output_size=model.fc2.out_features,
      bidirectional=model.bidirectional
    ).to(device)

    self.update_target()  # Copy weights instead of reinitializing

    self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
    self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.5)
    self.criterion = nn.MSELoss()
    self.batch_size = batch_size
    self.target_update_freq = target_update_freq
    self.train_step_count = 9

  def update_target(self):
    """Copy model weights to the target network"""
    self.target_model.load_state_dict(self.model.state_dict())  # ✅ Fixed

  def train_step(self, state, action, reward, next_state, done):
    state = state.to(device)
    next_state = next_state.to(device)
    action = action.to(device).view(-1, 1)  # Ensure proper shape for indexing
    reward = reward.to(device)
    done = done.to(device)

    pred = self.model(state)
    target = pred.clone().detach()

    with torch.no_grad():
      Q_next = self.target_model(next_state).max(dim=1)[0]  # Max Q-value for next state

    # Compute target Q-values
    target.scatter_(1, action, reward + (1 - done.float()) * self.gamma * Q_next.unsqueeze(1))

    # Compute loss and optimize
    self.optimizer.zero_grad()
    loss = self.criterion(pred, target)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)  # Gradient clipping
    self.optimizer.step()

    # Update target network every `target_update_freq` steps
    self.train_step_count += 1
    if self.train_step_count % self.target_update_freq == 0:
      self.update_target()

    # Adjust learning rate
    self.scheduler.step()
