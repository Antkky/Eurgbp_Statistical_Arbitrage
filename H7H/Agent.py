from Environment import *
from Model import *
import torch # type: ignore
import random
import pandas as pd # type: ignore
from collections import deque
from sklearn.preprocessing import MinMaxScaler, StandardScaler # type: ignore
import tracemalloc
import sys


# Constants for memory and training
MAX_MEMORY = 100_000
BATCH_SIZE = 64
LR = 0.25

class Agent:
    def __init__(self, df, args=""):
        self.window_size = 30
        self.file = args
        self.n_trades = 0
        self.epsilon = 0.25
        self.gamma = 0.95
        self.input_size = 13
        self.probability_distrabution = None

        self.memory = deque(maxlen=MAX_MEMORY)

        self.model = LSTM_Q_Net(input_size=self.input_size, hidden_size=64, output_size=2).to(device) # Initializes  Neural Network
        
        if (len(args) > 1):
            self.model.load_state_dict(torch.load(self.file, weights_only=True))

        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma, BATCH_SIZE=BATCH_SIZE) # Initializes         Trainer
        self.env = Environment(df=df, window_size=self.window_size)                         # Initializes     Environment
        
        self.buy_signals = []
        self.sell_signals = []

    def get_action(self, state):

        # Extract only numeric data
        if isinstance(state, pd.DataFrame):
            numeric_state = state.select_dtypes(include=[np.number]).values
        elif isinstance(state, pd.Series):
            numeric_state = state.values
        else:
            raise ValueError("State must be a Pandas DataFrame or Series.")

        numeric_state = np.nan_to_num(numeric_state, nan=0.0)

        state_tensor = torch.tensor(numeric_state, dtype=torch.float32).unsqueeze(0).to(device)
        q_values = self.model.forward(state_tensor)
        self.probability_distrabution = q_values
        print(torch.tensor(q_values).numpy())
        return torch.argmax(q_values).item()
            
    def store_experience(self, state, action, reward, next_state, done):
        clipped_reward = np.clip(reward, -1, 1)
        self.memory.append((state, action, clipped_reward, next_state, done))

    def sample_prioritized(self):
        # Calculate absolute rewards as priorities
        priorities = np.array([abs(self.memory[i][2]) for i in range(len(self.memory))])
        # Convert NaNs or Infs to a small number (e.g., 1e-6 to avoid division by zero)
        priorities = np.nan_to_num(priorities, nan=1e-6, posinf=1e-6, neginf=1e-6)
        # If the sum of priorities is zero (rare but possible), assign uniform probability
        if np.sum(priorities) == 0:
            probs = np.ones_like(priorities) / len(priorities)  # Uniform distribution
        else:
            probs = priorities / np.sum(priorities)  # Normalize to get probabilities
        # Sample indices based on calculated probabilities
        idxs = np.random.choice(len(self.memory), BATCH_SIZE, p=probs)
        mini_batch = [self.memory[i] for i in idxs]
        return mini_batch

    def train(self):
        if len(self.memory) < BATCH_SIZE:
            return  # Not enough experiences to sample

        #mini_batch = random.sample(self.memory, BATCH_SIZE)
        mini_batch = self.sample_prioritized()
        states, actions, rewards, next_states, dones = zip(*mini_batch)

        # Convert to Tensors and reshape
        states = torch.tensor(np.array(states), dtype=torch.float32).to(device).view(BATCH_SIZE, self.window_size, self.input_size)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(device).view(BATCH_SIZE, self.window_size, self.input_size)
        actions = torch.tensor(actions, dtype=torch.long).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        dones = torch.tensor(dones, dtype=torch.bool).to(device)

        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def update_epsilon(self):
        self.epsilon = max(0.01, self.epsilon * 0.99)  # Faster decay

    def save(self, episode):
        name = "Episode-" + str(episode) + ".pth"
        self.model.save(name)

    def run(self, episodes):
        for episode in range(episodes):
            print(f"Episode {episode + 1}")
            state = self.env.reset()
            done = False
            total_reward = 0
            while not done:
                action0 = self.get_action(state)
                match action0:
                    case 0:
                        action1 = Actions.Buy
                    case 1:
                        action1 = Actions.Sell
                
                state, reward, realprofit, unrealprofit, done = self.env.forward(action1)  # Interact with the environment
                self.store_experience(state.values, action0, reward, state.values, done)  # Store experience
                self.train()  # Train the agent
                total_reward += reward
            
            self.env.close()
            self.update_epsilon()
            self.save(episode)
            print(f"Episode: {episode}\nReward: {total_reward}\nProfit: {unrealprofit}")


if __name__ == "__main__":
    df = pd.read_csv('trade_data.csv')
    
    features = df.drop(columns=["Time"])

    min_max_scaler = MinMaxScaler()
    scaled_minmax = min_max_scaler.fit_transform(features)

    df_minmax = pd.DataFrame(scaled_minmax, columns=features.columns)

    if(len(sys.argv) > 1):
        arg = sys.argv[1]
        
    else:
        arg = ""

    agent = Agent(df_minmax, arg)
    agent.run(episodes=50)