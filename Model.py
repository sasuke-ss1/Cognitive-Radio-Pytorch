import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, state_size=4, action_size=2, hidden_size=10, step_size=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.action_size = action_size
        self.state_size = state_size

        self.lstm = nn.LSTM(input_size=state_size, hidden_size=hidden_size)
        self.linear1 = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.norm = nn.LayerNorm(hidden_size)
        self.linear2 = nn.Linear()

    def forward(self, x):
        h, c = self.lstm(x)
        self.out = self.norm(self.relu(self.linear1(h)))
        self.out = self.linear2(self.out)

        return self.out
    
