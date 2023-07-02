import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    def __init__(self, input_layer_dims, hidden_layer_dims, output_layer_dims):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(input_layer_dims, hidden_layer_dims)
        self.layer2 = nn.Linear(hidden_layer_dims, hidden_layer_dims)
        self.layer3 = nn.Linear(hidden_layer_dims, output_layer_dims)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)
        return x
