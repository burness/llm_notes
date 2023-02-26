import torch
import copy
import torch.nn as nn
from layer_norm import LayerNorm
import torch.nn.functional as F
from clones import clones

class Encoder(nn.Module):
    "core encoder is a stack of N layers"
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        "pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)