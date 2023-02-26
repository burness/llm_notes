import torch
import torch.nn as nn
from clones import clones
from attention import attention


class MultiHeadAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "take in model size and number of heads"
        super(MultiHeadAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
        x, self.attn = attention(query, key, value, mask=mask, 
                                 dropout=self.dropout)
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h*self.d_k)
        return self.linears[-1](x)