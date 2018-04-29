import torch
import torch.nn as nn
import torch.nn.functional as F

char_limit = 16
char_dim = 8
char_hidden_size = 100
char_num_layers = 1
char_direc = 2

dropout = 0.2
batch_size = 32
d_model = 128
h = 8
d_k = d_model // h
d_v = d_model // h
word_emb_size = 300
char_emb_size = char_direc * char_num_layers * char_hidden_size
emb_size = word_emb_size + char_emb_size

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Using bidirectional gru hidden state to represent char embedding for a word
class CharEmbedding(nn.Module):
    def __init__(self, in_size=word_emb_size):
        super(CharEmbedding, self).__init__()
        self.num_layers = 1
        self.bidirectional = True
        self.dir = 2 if self.bidirectional else 1
        self.hidden_size = char_hidden_size
        self.in_size = in_size
        self.gru = nn.GRU(input_size=in_size, bidirectional=self.bidirectional, num_layers=self.num_layers,
                          hidden_size=self.hidden_size)
        self.h = torch.randn(self.num_layers * self.dir, 1, self.hidden_size)
        self.out_size = self.hidden_size * self.num_layers * self.dir

    def forward(self, input):
        (l, b, in_size) = input.size()
        assert in_size == self.in_size and b == 1
        o, h = self.gru(input, self.h)
        h = h.view(-1)
        return h


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.depthwise_conv = nn.Conv1d(in_channels=in_ch, out_channels=in_ch, kernel_size=7, groups=in_ch,
                                        padding=3)
        self.pointwise_conv = nn.Conv1d(in_channels=in_ch, out_channels=out_ch, kernel_size=1, padding=0)

    def forward(self, x):
        return self.pointwise_conv(self.depthwise_conv(x))

class SelfAttention(nn.Module):
    def __init__(self):
        pass

    def forward(self, x):
        pass

class EncoderBlock(nn.Module):
    def __init__(self, conv_num: int, in_ch: int, out_ch: int):
        if in_ch != out_ch:
            self.conv_prime = DepthwiseSeparableConv(in_ch, out_ch)
        else:
            self.conv_prime = None
        self.convs = nn.ModuleList([DepthwiseSeparableConv(out_ch, out_ch) for i in range(conv_num)])
        self.self_att = SelfAttention()
        self.forward = nn.Linear(out_ch, out_ch)

    def norm(self, x, eps=1e-6):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return (x - mean) / (std + eps)


class StandardQANet(nn.Module):
    def __init__(self):
        pass
