# dqn_agent.py
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque

# В dqn_agent.py, ДОБАВЬТЕ НОВЫЙ КЛАСС:

class LSTMDQN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=64, num_layers=1):
        super(LSTMDQN, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        # x shape: (batch, seq_len, input_dim)
        lstm_out, _ = self.lstm(x)
        # Берём последний шаг последовательности
        last_output = lstm_out[:, -1, :]
        return self.fc(last_output)

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.fc(x)

class LSTMDQNAgent:
    def __init__(self, state_size, action_size, sequence_length=5, lr=0.001, gamma=0.95, 
                 epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, memory_size=2000, batch_size=32):
        self.state_size = state_size
        self.action_size = action_size
        self.sequence_length = sequence_length
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.memory = deque(maxlen=memory_size)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.q_network = LSTMDQN(state_size, action_size).to(self.device)
        self.target_network = LSTMDQN(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.update_target()

    def update_target(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def remember(self, state_sequence, action, reward, next_state_sequence, done):
        self.memory.append((state_sequence, action, reward, next_state_sequence, done))

    def act(self, state_history):
        # state_history: list of recent states, length <= sequence_length
        if len(state_history) < self.sequence_length:
            # Дополняем нулями спереди
            padding = [np.zeros(self.state_size) for _ in range(self.sequence_length - len(state_history))]
            state_seq = padding + state_history
        else:
            state_seq = state_history[-self.sequence_length:]
        
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        state_tensor = torch.FloatTensor([state_seq]).to(self.device)  # (1, seq_len, state_size)
        q_values = self.q_network(state_tensor)
        return q_values.cpu().detach().numpy().argmax()

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        
        state_seqs = torch.FloatTensor([e[0] for e in batch]).to(self.device)
        actions = torch.LongTensor([e[1] for e in batch]).to(self.device)
        rewards = torch.FloatTensor([e[2] for e in batch]).to(self.device)
        next_state_seqs = torch.FloatTensor([e[3] for e in batch]).to(self.device)
        dones = torch.BoolTensor([e[4] for e in batch]).to(self.device)

        current_q = self.q_network(state_seqs).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q = self.target_network(next_state_seqs).max(1)[0].detach()
        target_q = rewards + (self.gamma * next_q * ~dones)

        loss = nn.MSELoss()(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay