import torch
from torch import nn


class DynamicsNet(nn.Module):
    """Small MLP. Defaults (3 hidden layers x 64, SiLU) reproduce the archived architecture,
    including its state_dict keys, so outputs/checkpoints/dynamics_synthetic.pt still loads."""

    def __init__(self, inputs=10, outputs=5, hidden_width=64, hidden_layers=3):
        super().__init__()
        layers, width = [], inputs
        for _ in range(hidden_layers):
            layers += [nn.Linear(width, hidden_width), nn.SiLU()]
            width = hidden_width
        layers.append(nn.Linear(width, outputs))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
