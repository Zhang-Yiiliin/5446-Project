import torch
import torch.nn as nn

import random

class MLP_DQN(nn.Module):
    def __init__(self, num_inputs, num_actions):
        super(MLP_DQN, self).__init__()

        self.num_inputs = num_inputs
        self.num_actions = num_actions
        
        self.layers = nn.Sequential(
            nn.Linear(num_inputs, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions)
        )

    def forward(self, x):
        return self.layers(x)
    
    def act(self, state, epsilon):
        """ select an action """
        if random.random() > epsilon:
            # exploit
            state = torch.FloatTensor(state).unsqueeze(0)
            q_value = self.forward(state)
            action  = q_value.max(1)[1].item()
        else:
            # explore
            action = random.randrange(self.num_actions)
        return action
