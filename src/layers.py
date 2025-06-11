
import torch
import torch.nn as nn
import numpy as np


class WeightedResidualConnection(nn.Module):
    """
    y = inputs + α * residual     with learnable scalar α
    """
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))
    
    def forward(self, inputs: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        return inputs + self.weight * residual


class ECALayer(nn.Module):
    """
    Efficient Channel Attention (ECA) 
    """
    def __init__(self, channels: int, gamma: int = 2, b: int = 1):
        super().__init__()
        t = abs((np.log(channels) / np.log(2.0) + b) / gamma)
        k = int(t) if int(t) % 2 else int(t) + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)          
        self.conv1d   = nn.Conv1d(1, 1, kernel_size=k,
                                  padding=k // 2, bias=True)
        self.sigmoid  = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.avg_pool(x)                             
        y = y.squeeze(-1).transpose(1, 2)                
        y = self.conv1d(y)                               
        y = self.sigmoid(y)
        y = y.transpose(1, 2).unsqueeze(-1)             
        return x * y


