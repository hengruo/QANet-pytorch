import torch
import torch.nn as nn
import torch.nn.functional as F
import config

char_limit = config.char_limit
char_dim = config.char_dim
char_hidden_size = config.char_hidden_size
char_num_layers = config.char_num_layers
char_dir = config.char_dir

dropout = config.dropout
batch_size = config.batch_size
hidden_size = config.hidden_size
word_emb_size = config.word_emb_size
char_emb_size = config.char_emb_size
emb_size = config.emb_size

device = config.device

# Using bidirectional gru hidden state to represent char embedding for a word
class CharEmbedding(nn.Module):
    def __init__(self, in_size=word_emb_size):
        super(CharEmbedding, self).__init__()
        self.num_layers = 1
        self.bidirectional = True
        self.dir = 2 if self.bidirectional else 1
        self.hidden_size = char_hidden_size
        self.in_size = in_size
        self.gru = nn.GRU(input_size=in_size, bidirectional=self.bidirectional, num_layers=self.num_layers, hidden_size=self.hidden_size)
        self.h = torch.randn(self.num_layers*self.dir, 1, self.hidden_size)
        self.out_size = self.hidden_size * self.num_layers * self.dir

    def forward(self, input):
        (l, b, in_size) = input.size()
        assert in_size == self.in_size and b == 1
        o, h = self.gru(input, self.h)
        h = h.view(-1)
        return h

class NormLayer(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return (x - mean) / (std + self.eps)

class ConvLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = NormLayer()
        self.conv = nn.Conv1d()

class EncoderBlock(nn.Module):
    def __init__(self, conv_num: int):
        pass

class StandardQANet(nn.Module):
    def __init__(self):
        pass