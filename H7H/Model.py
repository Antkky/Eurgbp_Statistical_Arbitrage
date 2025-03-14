import torch # type: ignore
import torch.nn as nn # type: ignore
import torch.optim as optim # type: ignore
import torch.nn.functional as F # type: ignore
import numpy as np
import os

class LSTM_Q_Net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout=0.5, bidirectional=True):
        super(LSTM_Q_Net, self).__init__()
        
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, 
                            num_layers=5, batch_first=True, 
                            dropout=dropout, bidirectional=bidirectional)
        
        self.ln = nn.LayerNorm(hidden_size * 2 if bidirectional else hidden_size)
        
        # Add a few more fully connected layers for capacity
        self.fc1 = nn.Linear(hidden_size * 2 if bidirectional else hidden_size, 128)
        self.fc2 = nn.Linear(128, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[:, -1, :]  # Get the last time step's output
        lstm_out = self.ln(lstm_out)
        
        # Pass through additional layers
        x = F.relu(self.fc1(lstm_out))
        x = self.fc2(x)

        return x


    def save(self, file_name):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)


class QTrainer:
    def __init__(self, model, lr, gamma, BATCH_SIZE):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.target_model = LSTM_Q_Net(13, 64, 2) # Target model
        self.update_target()
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.5)
        self.criterion = nn.MSELoss()
        self.batch_size = BATCH_SIZE

    def update_target(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def train_step(self, state, action, reward, next_state, done):
        # Ensure correct shape for LSTM (batch_size, sequence_length, input_size)
        if state.dim() == 2:
            state = state.view(-1, 30, 13)
        if next_state.dim() == 2:
            next_state = next_state.view(-1, 30, 13)


        pred = self.model(state)
        target = pred.clone()

        for idx in range(len(done)):
            with torch.no_grad():
                Q_next = self.target_model(next_state[idx].unsqueeze(0)).max()
            if done[idx]:
                target[idx][action[idx]] = reward[idx]
            else:
                target[idx][action[idx]] = reward[idx] + self.gamma * Q_next

        self.optimizer.zero_grad()
        loss = self.criterion(pred, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)  # Gradient clipping
        self.optimizer.step()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

