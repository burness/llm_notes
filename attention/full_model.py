import copy
import torch.nn as nn
from multi_head_attention import MultiHeadAttention
from pw_ffn import PositionwiseFeedForward
from positional_encoding import PositionalEncoding
from model_arch import EncoderDecoder
from encoder import Encoder
from encoder_layer import EncoderLayer
from decoder import Decoder
from decoder_layer import DecoderLayer
from embedding import Embeddings
from generator import Generator


def make_model(src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
    "constuct a model from hyperparameters"
    c = copy.deepcopy
    attn = MultiHeadAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab)
    )
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)
    return model
