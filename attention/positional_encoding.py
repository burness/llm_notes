import torch
import math
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
import matplotlib.pyplot as plt


class PositionalEncoding(nn.Module):
    "implement the positional encoding function."
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-1.0*math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:,:x.size(1)], requires_grad=False)
        return self.dropout(x)


if __name__ == "__main__":
    plt.figure(figsize=(15, 5))
    pe = PositionalEncoding(20, 0)
    y = pe.forward(Variable(torch.zeros(1, 100, 20)))
    plt.plot(np.arange(100), y[0, :, 4:8].data.numpy())
    plt.legend(["dim %d"%p for p in [4,5,6,7]])
    plt.savefig("./pe_demo.jpg")