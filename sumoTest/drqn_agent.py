# drqn_agent.py (улучшенная версия)
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque

class DRQN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128, num_layers=1):
        super(DRQN, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x, hidden=None):
        lstm_out, hidden = self.lstm(x, hidden)
        q_values = self.fc(lstm_out[:, -1, :])
        return q_values, hidden

    def init_hidden(self, batch_size=1, device="cpu"):
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        return (h0, c0)


class DRQNAgent:
    def __init__(self, state_size, action_size, sequence_length=5,
                 lr=0.0005, gamma=0.99, epsilon=1.0, epsilon_min=0.05,
                 epsilon_decay=0.98, memory_size=5000, batch_size=64):
        self.state_size = state_size
        self.action_size = action_size
        self.seq_len = sequence_length
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.memory = deque(maxlen=memory_size)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_network = DRQN(state_size, action_size, hidden_dim=128).to(self.device)
        self.target_network = DRQN(state_size, action_size, hidden_dim=128).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.update_target()

    def update_target(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def remember(self, sequence, action, reward, next_sequence, done):
        self.memory.append((sequence.copy(), action, reward, next_sequence.copy(), done))

    def act(self, state_sequence):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state_tensor = torch.FloatTensor(state_sequence).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values, _ = self.q_network(state_tensor)
        return q_values.cpu().argmax().item()

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        sequences = torch.FloatTensor([e[0] for e in batch]).to(self.device)
        actions = torch.LongTensor([e[1] for e in batch]).to(self.device)
        rewards = torch.FloatTensor([e[2] for e in batch]).to(self.device)
        next_sequences = torch.FloatTensor([e[3] for e in batch]).to(self.device)
        dones = torch.BoolTensor([e[4] for e in batch]).to(self.device)

        current_q, _ = self.q_network(sequences)
        current_q = current_q.gather(1, actions.unsqueeze(1)).squeeze(1)

        next_q, _ = self.target_network(next_sequences)
        next_q = next_q.max(1)[0].detach()

        target_q = rewards + (self.gamma * next_q * ~dones)

        loss = nn.MSELoss()(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        # 🔑 Градиентный клиппинг — ключ к стабильности!
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay