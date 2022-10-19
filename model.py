from turtle import forward
from sympy import Rel
import torch.nn.functional as F
import torch.nn as nn
import torch
class CNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embadding=nn.Sequential(

        )
        self.FC=nn.Sequential(
            nn.Linear(),
            F.relu(),
            nn.Linear(),
            F.relu()
        )
    def forward():
        pass
class SNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.FC=nn.Sequential(
            nn.Linear(),
            F.relu(),
            nn.Linear(),
            F.relu(),
            nn.Linear(),
            nn.Softmax(dim=1)
        )
    def forward():
        pass
class Steal(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    def forward():
        pass