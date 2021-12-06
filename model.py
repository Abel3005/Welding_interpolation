import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3,3)
        
    def forward(self, x):
        return self.linear(x)
