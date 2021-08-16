import torch
from torch import nn


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(10, 1)
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits
