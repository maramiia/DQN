# dqn_agent.py
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque

# ==================== ПРОСТОЙ DQN ====================
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

class DQNAgent:
    def __init__(self, state_size, action_size, lr=0.001, gamma=0.95, 
                 epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, 
                 memory_size=10000, batch_size=32):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.memory = deque(maxlen=memory_size)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_network = DQN(state_size, action_size).to(self.device)
        self.target_network = DQN(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.update_target()

    def update_target(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        
        return q_values.cpu().numpy().argmax()

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
            
        batch = random.sample(self.memory, self.batch_size)
        
        states = torch.FloatTensor([e[0] for e in batch]).to(self.device)
        actions = torch.LongTensor([e[1] for e in batch]).to(self.device)
        rewards = torch.FloatTensor([e[2] for e in batch]).to(self.device)
        next_states = torch.FloatTensor([e[3] for e in batch]).to(self.device)
        dones = torch.BoolTensor([e[4] for e in batch]).to(self.device)

        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q = self.target_network(next_states).max(1)[0].detach()
        target_q = rewards + (self.gamma * next_q * ~dones)

        loss = nn.MSELoss()(current_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, filename):
        torch.save(self.q_network.state_dict(), filename)
        
    def load(self, filename):
        self.q_network.load_state_dict(torch.load(filename, map_location=self.device))
        self.update_target()


# ==================== DQN с LSTM ====================
class LSTMDQN(nn.Module):  # ← ИЗМЕНИЛ НАЗВАНИЕ НА LSTMDQN
    def __init__(self, input_dim, output_dim, hidden_size=64, num_layers=1):
        super(LSTMDQN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_dim, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
        
    def forward(self, x, hidden=None):
        batch_size = x.size(0)
        
        if hidden is None:
            h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
            c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
            hidden = (h0, c0)
            
        lstm_out, hidden = self.lstm(x, hidden)
        last_output = lstm_out[:, -1, :]
        q_values = self.fc(last_output)
        
        return q_values, hidden

class LSTMDQNAgent:  # ← ИЗМЕНИЛ НАЗВАНИЕ НА LSTMDQNAgent
    def __init__(self, state_size, action_size, sequence_length=5, lr=0.0005, gamma=0.99,
                 epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.998,
                 memory_size=10000, batch_size=64):
        self.state_size = state_size
        self.action_size = action_size
        self.sequence_length = sequence_length
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.memory = deque(maxlen=memory_size)
        
        self.sequence_memory = deque(maxlen=memory_size)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_network = LSTMDQN(state_size, action_size).to(self.device)
        self.target_network = LSTMDQN(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr, weight_decay=1e-5)
        self.update_target()
        
        self.current_sequence = deque(maxlen=sequence_length)

    def update_target(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def remember_sequence(self, state, action, reward, next_state, done):
        if len(self.current_sequence) < self.sequence_length:
            while len(self.current_sequence) < self.sequence_length:
                self.current_sequence.append(state)
        else:
            self.current_sequence.append(state)
            
        if len(self.current_sequence) == self.sequence_length:
            sequence_state = list(self.current_sequence)
            self.sequence_memory.append((sequence_state, action, reward, next_state, done))

    def act(self, state):
        self.current_sequence.append(state)
        
        if len(self.current_sequence) < self.sequence_length:
            return random.randrange(self.action_size)
            
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        sequence = list(self.current_sequence)
        state_tensor = torch.FloatTensor([sequence]).to(self.device)
        
        with torch.no_grad():
            q_values, _ = self.q_network(state_tensor)
        
        return q_values.cpu().numpy().argmax()

    def replay(self):
        if len(self.sequence_memory) < self.batch_size:
            return
            
        batch = random.sample(self.sequence_memory, self.batch_size)
        
        states = np.array([item[0] for item in batch])
        actions = np.array([item[1] for item in batch])
        rewards = np.array([item[2] for item in batch])
        
        next_states_list = []
        for item in batch:
            current_seq = item[0]
            next_state = item[3]
            new_sequence = current_seq[1:] + [next_state]
            next_states_list.append(new_sequence)
        
        next_states = np.array(next_states_list)
        dones = np.array([item[4] for item in batch])
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)
        
        current_q, _ = self.q_network(states)
        current_q = current_q.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        next_q, _ = self.target_network(next_states)
        next_q = next_q.max(1)[0].detach()
        target_q = rewards + (self.gamma * next_q * ~dones)
        
        loss = nn.MSELoss()(current_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def reset_sequence(self):
        self.current_sequence.clear()

    def save(self, filename):
        torch.save(self.q_network.state_dict(), filename)
        
    def load(self, filename):
        self.q_network.load_state_dict(torch.load(filename, map_location=self.device))
        self.update_target()