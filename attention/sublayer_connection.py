import torch
import torch.nn as nn
from layer_norm import LayerNorm

class SubLayerConnection(nn.Module):
    """
    a residual connect followed by a layer norm.
    """
    def __init__(self, size, dropout):
        super(SubLayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))