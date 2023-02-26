import torch
import torch.nn as nn
import copy

def clones(module, N):
    "produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])