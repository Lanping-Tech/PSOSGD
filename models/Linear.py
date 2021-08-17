import torch
from torch import nn

class Config:

    """模型配置参数"""
    def __init__(self):
        self.model_lib = 'models.Linear'
        self.input_dim = 10
        self.output_dim = 1


class Model(nn.Module):
    def __init__(self, input_dim, output_dim, **kwargs):
        super(Model, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_dim, output_dim)
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits
