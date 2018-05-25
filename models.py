import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from config import config, device, cpu

conn_dim = config.connector_dim
num_head = config.num_heads
word_dim = config.glove_dim
char_dim = config.char_dim
batch_size = config.batch_size
dropout = config.dropout
dropout_char = config.dropout_char

d_k = conn_dim // num_head
d_v = conn_dim // num_head
cq_att_size = conn_dim * 4


def norm(x, eps=1e-6):
    mean = x.mean(-1, keepdim=True)
    std = x.std(-1, keepdim=True)
    return (x - mean) / (std + eps)


class PosEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        freqs = torch.Tensor([10000 ** (-i / conn_dim) if i % 2 == 0 else -10000 ** (-(i - 1) / conn_dim) for i in
                                   range(conn_dim)]).unsqueeze(dim=1)
        phases = torch.Tensor([0 if i % 2 == 0 else math.pi / 2 for i in range(conn_dim)]).unsqueeze(dim=1)
        para_pos = torch.arange(config.para_limit).repeat(conn_dim, 1)
        ques_pos = torch.arange(config.ques_limit).repeat(conn_dim, 1)
        self.para_pos_encoding = torch.sin(torch.add(torch.mul(para_pos, freqs), phases)).to(device)
        self.ques_pos_encoding = torch.sin(torch.add(torch.mul(ques_pos, freqs), phases)).to(device)

    def forward(self, x):
        (_, _, l) = x.size()
        if l == config.para_limit:
            out = self.para_pos_encoding + x
        elif l == config.ques_limit:
            out = self.ques_pos_encoding + x
        else:
            raise ValueError("Wrong data length. Neither {} nor {}".format(config.para_limit, config.ques_limit))
        return out


# Using bidirectional gru hidden state to represent char embedding for a word
class CharEmbedding(nn.Module):
    def __init__(self, in_size=word_dim):
        super().__init__()
        self.num_layers = 1
        self.bidirectional = True
        self.dir = 2 if self.bidirectional else 1
        self.hidden_size = 100
        self.in_size = in_size
        self.gru = nn.GRU(input_size=in_size, bidirectional=self.bidirectional, num_layers=self.num_layers,
                          hidden_size=self.hidden_size)
        self.h = torch.randn(self.num_layers * self.dir, 1, self.hidden_size)
        self.out_size = self.hidden_size * self.num_layers * self.dir

    def forward(self, input):
        o, h = self.gru(input, self.h)
        h = h.view(-1)
        return h


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch, k, dim=1, bias=False):
        super().__init__()
        if dim == 1:
            self.depthwise_conv = nn.Conv1d(in_channels=in_ch, out_channels=in_ch, kernel_size=k, groups=in_ch,
                                            padding=k // 2, bias=bias)
            self.pointwise_conv = nn.Conv1d(in_channels=in_ch, out_channels=out_ch, kernel_size=1, padding=0, bias=bias)
        elif dim == 2:
            self.depthwise_conv = nn.Conv2d(in_channels=in_ch, out_channels=in_ch, kernel_size=k, groups=in_ch,
                                            padding=k // 2, bias=bias)
            self.pointwise_conv = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=1, padding=0, bias=bias)
        else:
            raise Exception("Wrong dimension for Depthwise Separable Convolution!")

    def forward(self, x):
        return self.pointwise_conv(self.depthwise_conv(x))


class Highway(nn.Module):
    def __init__(self, layer_num: int, size=conn_dim):
        super().__init__()
        self.n = layer_num
        self.linear = nn.ModuleList([nn.Linear(size, size) for _ in range(self.n)])
        self.gate = nn.ModuleList([nn.Linear(size, size) for _ in range(self.n)])
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.transpose(1, 2)
        for i in range(self.n):
            gate = F.sigmoid(self.gate[i](x))
            nonlinear = self.relu(self.linear[i](x))
            x = gate * nonlinear + (1 - gate) * x
        x = x.transpose(1, 2)
        return x


class SelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.Wqs = [torch.empty(batch_size, d_k, conn_dim, device=device, requires_grad=True) for _ in range(num_head)]
        self.Wks = [torch.empty(batch_size, d_k, conn_dim, device=device, requires_grad=True) for _ in range(num_head)]
        self.Wvs = [torch.empty(batch_size, d_v, conn_dim, device=device, requires_grad=True) for _ in range(num_head)]
        self.Wo = torch.empty(batch_size, d_v * num_head, conn_dim, device=device, requires_grad=True)
        nn.init.xavier_normal_(self.Wo)
        for i in range(num_head):
            nn.init.xavier_normal_(self.Wqs[i])
            nn.init.xavier_normal_(self.Wks[i])
            nn.init.xavier_normal_(self.Wvs[i])

    def forward(self, x: torch.Tensor):
        WQs, WKs, WVs = [], [], []
        for i in range(num_head):
            WQs.append(torch.bmm(self.Wqs[i], x))
            WKs.append(torch.bmm(self.Wks[i], x))
            WVs.append(torch.bmm(self.Wvs[i], x))
        heads = []
        for i in range(num_head):
            out = torch.bmm(WQs[i].transpose(1, 2), WKs[i])
            out = torch.div(out, math.sqrt(d_k))
            # not sure... I think `dim` should be 1 since it weighted each column of `WVs[i]`
            out = F.softmax(out, dim=1)
            headi = torch.bmm(WVs[i], out)
            heads.append(headi)
        head = torch.cat(heads, dim=1)
        out = torch.bmm(self.Wo, head)
        return out


class Embedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.drop_c = nn.Dropout(p=dropout_char)
        self.conv2d = DepthwiseSeparableConv(char_dim, conn_dim, 5, dim=2, bias=True)
        self.relu = nn.ReLU()
        self.conv1d = DepthwiseSeparableConv(word_dim + conn_dim, conn_dim, 5, bias=True)
        self.drop_w = nn.Dropout(p=dropout)
        self.high = Highway(2)

    def forward(self, ch_emb, wd_emb):
        ch_emb = ch_emb.permute(0, 3, 1, 2)
        ch_emb = self.drop_c(ch_emb)
        ch_emb = self.conv2d(ch_emb)
        ch_emb = self.relu(ch_emb)
        ch_emb, _ = torch.max(ch_emb, dim=3)
        ch_emb = ch_emb.squeeze()
        wd_emb = self.drop_w(wd_emb)
        wd_emb = wd_emb.transpose(1, 2)
        emb = torch.cat([ch_emb, wd_emb], dim=1)
        emb = self.conv1d(emb)
        emb = self.high(emb)
        return emb


class EncoderBlock(nn.Module):
    def __init__(self, conv_num: int, ch_num: int, k: int):
        super().__init__()
        self.convs = nn.ModuleList([DepthwiseSeparableConv(ch_num, ch_num, k) for _ in range(conv_num)])
        self.self_att = SelfAttention()
        self.W = torch.empty(batch_size, ch_num, ch_num, device=device, requires_grad=True)
        nn.init.xavier_normal_(self.W)
        self.pos = PosEncoder()
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.pos(x)
        res = out
        for i, conv in enumerate(self.convs):
            out = norm(out)
            out = conv(out)
            out = self.relu(out)
            out = res + out
            if (i + 1) % 2 == 0:
                out = F.dropout(out, p=dropout, training=True)
            res = out
        out = norm(out)
        out = self.self_att(out)
        out = res + out
        out = F.dropout(out, p=dropout, training=True)
        res = out
        out = norm(out)
        out = torch.bmm(self.W, out)
        out = self.relu(out)
        out = res + out
        out = F.dropout(out, p=dropout, training=True)
        return out


class CQAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.W = torch.empty(batch_size, 1, conn_dim * 3, device=device, requires_grad=True)
        nn.init.xavier_normal_(self.W)
        self.S = torch.zeros(batch_size, config.para_limit, config.ques_limit, device=device)

    def forward(self, C, Q):
        for i in range(config.para_limit):
            for j in range(config.ques_limit):
                c = C[:, :, i]
                q = Q[:, :, j]
                v = torch.cat([q, c, torch.mul(q, c)], dim=1).unsqueeze(dim=2)
                self.S[:, i, j] = torch.bmm(self.W, v).squeeze()
        S_ = F.softmax(self.S, dim=2)
        S__ = F.softmax(self.S, dim=1)
        A = torch.bmm(Q, S_.transpose(1, 2))
        B = torch.bmm(C, torch.bmm(S_, S__.transpose(1, 2)).transpose(1, 2))
        out = torch.cat([C, A, torch.mul(C, A), torch.mul(C, B)], dim=1)
        out = F.dropout(out, p=dropout, training=True)
        return out


class Pointer(nn.Module):
    def __init__(self):
        super().__init__()
        self.W0 = torch.empty(batch_size, 1, conn_dim * 2, device=device, requires_grad=True)
        self.W1 = torch.empty(batch_size, 1, conn_dim * 2, device=device, requires_grad=True)
        nn.init.xavier_normal_(self.W0)
        nn.init.xavier_normal_(self.W1)

    def forward(self, M0, M1, M2):
        X0 = torch.cat([M0, M1], dim=1)
        X1 = torch.cat([M0, M2], dim=1)
        Y0 = torch.bmm(self.W0, X0).squeeze()
        Y1 = torch.bmm(self.W1, X1).squeeze()
        p0 = F.log_softmax(Y0, dim=1)
        p1 = F.log_softmax(Y1, dim=1)
        return p0, p1


class QANet(nn.Module):
    def __init__(self, word_mat, char_mat):
        super().__init__()
        self.char_emb = nn.Embedding.from_pretrained(torch.Tensor(char_mat), freeze=(not config.pretrained_char))
        self.word_emb = nn.Embedding.from_pretrained(torch.Tensor(word_mat))
        self.emb = Embedding()
        self.c_emb_enc = EncoderBlock(conv_num=4, ch_num=conn_dim, k=7)
        self.q_emb_enc = EncoderBlock(conv_num=4, ch_num=conn_dim, k=7)
        self.cq_att = CQAttention()
        self.cq_resizer = DepthwiseSeparableConv(conn_dim * 4, conn_dim, 5)
        self.enc_num = 3
        self.model_enc_blk0 = EncoderBlock(conv_num=2, ch_num=conn_dim, k=5)
        self.model_enc_blk1 = EncoderBlock(conv_num=2, ch_num=conn_dim, k=5)
        self.model_enc_blk2 = EncoderBlock(conv_num=2, ch_num=conn_dim, k=5)
        self.out = Pointer()

    def move(self, device):
        self.to(device)


    def forward(self, Cwid, Ccid, Qwid, Qcid):
        Cw, Cc = self.word_emb(Cwid), self.char_emb(Ccid)
        Qw, Qc = self.word_emb(Qwid), self.char_emb(Qcid)
        C, Q = self.emb(Cc, Cw), self.emb(Qc, Qw)
        C = self.c_emb_enc(C)
        Q = self.q_emb_enc(Q)
        X = self.cq_att(C, Q)
        X = self.cq_resizer(X)
        M0 = X
        for i in range(7):
            M0 = self.model_enc_blk0(M0)
        M1 = M0
        for i in range(7):
            M1 = self.model_enc_blk1(M1)
        M2 = M1
        for i in range(7):
            M2 = self.model_enc_blk2(M2)
        p1, p2 = self.out(M0, M1, M2)
        p1, p2 = torch.exp(p1), torch.exp(p2)
        return p1, p2
