import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    def __init__(self, input_dim: int, out_dim: int, init_w: float = 3e-3, nn_layers: list = [300, 200], dropout_rate: float = 0.5):
        """Initialize."""
        super(Actor, self).__init__()
        self.nn_layers = nn_layers

        layers = []
        for i in range(len(nn_layers)):
            in_features = input_dim if i == 0 else nn_layers[i-1]
            layers.append(nn.Linear(in_features, nn_layers[i]))
            nn.init.kaiming_normal_(layers[-1].weight, nonlinearity='relu')  # He Initialization
            layers.append(nn.LeakyReLU())

        self.layers = nn.Sequential(*layers)
        self.out = nn.Linear(nn_layers[-1], out_dim)
        self.out.weight.data.uniform_(-init_w, init_w)
        self.out.bias.data.uniform_(-init_w, init_w)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        x = self.layers(state)
        action = torch.sigmoid(self.out(x))
        return action
    

class Critic(nn.Module):
    def __init__(self, input_dim: int, init_w: float = 3e-3, nn_layers: list = [300, 200], dropout_rate: float = 0.5):
        """Initialize."""
        super(Critic, self).__init__()
        self.nn_layers = nn_layers

        layers = []
        for i in range(len(nn_layers)):
            in_features = input_dim if i == 0 else nn_layers[i-1]
            layers.append(nn.Linear(in_features, nn_layers[i]))
            nn.init.kaiming_normal_(layers[-1].weight, nonlinearity='relu')  # He Initialization
            layers.append(nn.LeakyReLU())
            layers.append(nn.Dropout(dropout_rate))  # Dropout

        self.layers = nn.Sequential(*layers)
        self.out = nn.Linear(nn_layers[-1], 1)
        self.out.weight.data.uniform_(-init_w, init_w)
        self.out.bias.data.uniform_(-init_w, init_w)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        x = torch.cat((state, action), dim=-1)
        x = self.layers(x)
        value = self.out(x)
        return value
