import torch
import torch.nn as nn
import torch.nn.functional as F
import math

dropout = 0.1
dropout_w = 0.1
dropout_c = 0.05
batch_size = 24
d_model = 64
h = 8
d_k = d_model // h
d_v = d_model // h
cq_att_size = d_model * 4
word_emb_size = 300
char_emb_size = 200
emb_size = word_emb_size + char_emb_size
training = True
max_char_num = 16

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cpu = torch.device("cpu")

freqs = torch.Tensor(
    [10000 ** (-i / d_model) if i % 2 == 0 else -10000 ** (-(i - 1) / d_model) for i in range(d_model)]).unsqueeze(
    dim=1).to(device)
phases = torch.Tensor([0 if i % 2 == 0 else math.pi / 2 for i in range(d_model)]).unsqueeze(dim=1).to(device)


def norm(x, eps=1e-6):
    mean = x.mean(-1, keepdim=True)
    std = x.std(-1, keepdim=True)
    return (x - mean) / (std + eps)


def pos_encoding(x):
    (_, d, l) = x.size()
    pos = torch.arange(l).repeat(d, 1).to(device)
    tmp1 = torch.mul(pos, freqs)
    tmp2 = torch.add(tmp1, phases)
    pos_enc = torch.sin(tmp2)
    out = torch.sin(pos_enc) + x
    return out


# Using bidirectional gru hidden state to represent char embedding for a word
class CharEmbedding(nn.Module):
    def __init__(self, in_size=word_emb_size):
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
        (l, b, in_size) = input.size()
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
    def __init__(self, layer_num: int, size=d_model):
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
        self.Wqs = [torch.empty(batch_size, d_k, d_model, device=device, requires_grad=True) for _ in range(h)]
        self.Wks = [torch.empty(batch_size, d_k, d_model, device=device, requires_grad=True) for _ in range(h)]
        self.Wvs = [torch.empty(batch_size, d_v, d_model, device=device, requires_grad=True) for _ in range(h)]
        self.Wo = torch.empty(batch_size, d_v * h, d_model, device=device, requires_grad=True)
        nn.init.xavier_normal_(self.Wo)
        for i in range(h):
            nn.init.xavier_normal_(self.Wqs[i])
            nn.init.xavier_normal_(self.Wks[i])
            nn.init.xavier_normal_(self.Wvs[i])

    def forward(self, x: torch.Tensor):
        assert x.size()[1] == d_model
        WQs, WKs, WVs = [], [], []
        for i in range(h):
            WQs.append(torch.bmm(self.Wqs[i], x))
            WKs.append(torch.bmm(self.Wks[i], x))
            WVs.append(torch.bmm(self.Wvs[i], x))
        heads = []
        for i in range(h):
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
        self.drop_c = nn.Dropout(p=dropout_c)
        self.conv2d = DepthwiseSeparableConv(char_emb_size, d_model, 5, dim=2, bias=True)
        self.relu = nn.ReLU()
        self.conv1d = DepthwiseSeparableConv(word_emb_size + d_model, d_model, 5, bias=True)
        self.drop_w = nn.Dropout(p=dropout_w)
        self.high = Highway(2)

    def forward(self, ch_emb, wd_emb):
        (N, L, cn, sz) = ch_emb.size()
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
        self.relu = nn.ReLU()

    def forward(self, x):
        out = pos_encoding(x)
        res = out
        for i, conv in enumerate(self.convs):
            out = norm(out)
            out = conv(out)
            out = self.relu(out)
            out = res + out
            if (i + 1) % 2 == 0:
                out = F.dropout(out, p=dropout, training=training)
            res = out
        out = norm(out)
        out = self.self_att(out)
        out = res + out
        out = F.dropout(out, p=dropout, training=training)
        res = out
        out = norm(out)
        out = torch.bmm(self.W, out)
        out = self.relu(out)
        out = res + out
        out = F.dropout(out, p=dropout, training=training)
        return out


class CQAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.W = torch.empty(batch_size, 1, d_model * 3, device=device, requires_grad=True)
        nn.init.xavier_normal_(self.W)

    def forward(self, C, Q):
        (_, _, n) = C.size()
        (_, _, m) = Q.size()
        S = torch.zeros(batch_size, n, m, device=device)
        for i in range(n):
            for j in range(m):
                c = C[:, :, i]
                q = Q[:, :, j]
                v = torch.cat([q, c, torch.mul(q, c)], dim=1).unsqueeze(dim=2)
                S[:, i, j] = torch.bmm(self.W, v).squeeze()
        S_ = F.softmax(S, dim=2)
        S__ = F.softmax(S, dim=1)
        A = torch.bmm(Q, S_.transpose(1, 2))
        B = torch.bmm(C, torch.bmm(S_, S__.transpose(1, 2)).transpose(1, 2))
        out = torch.cat([C, A, torch.mul(C, A), torch.mul(C, B)], dim=1)
        out = F.dropout(out, p=dropout, training=training)
        return out


class Pointer(nn.Module):
    def __init__(self):
        super().__init__()
        self.W0 = torch.empty(batch_size, 1, d_model * 2, device=device, requires_grad=True)
        self.W1 = torch.empty(batch_size, 1, d_model * 2, device=device, requires_grad=True)
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
    def __init__(self, data):
        super().__init__()
        self.char_emb = nn.Embedding(128, char_emb_size)
        self.word_emb = nn.Embedding.from_pretrained(torch.Tensor(data.word_embedding))
        self.emb = Embedding()
        self.c_emb_enc = EncoderBlock(conv_num=4, ch_num=d_model, k=7)
        self.q_emb_enc = EncoderBlock(conv_num=4, ch_num=d_model, k=7)
        self.cq_att = CQAttention()
        self.cq_resizer = DepthwiseSeparableConv(d_model * 4, d_model, 5)
        self.enc_num = 3
        self.model_enc_blk0 = EncoderBlock(conv_num=2, ch_num=d_model, k=5)
        self.model_enc_blk1 = EncoderBlock(conv_num=2, ch_num=d_model, k=5)
        self.model_enc_blk2 = EncoderBlock(conv_num=2, ch_num=d_model, k=5)
        self.out = Pointer()

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
        p0, p1 = self.out(M0.to(device), M1.to(device), M2)
        p0, p1 = torch.exp(p0), torch.exp(p1)
        return p0, p1


if __name__ == "__main__":
    import dataset

    squad = dataset.SQuAD.load("data/")
    model = QANet(squad)
    import random

    wl = list(range(1200))
    cl = list(range(50))
    Cw = torch.LongTensor(random.sample(wl, 150)).repeat(batch_size, 1)
    Cc = torch.LongTensor(random.sample(cl, 16)).repeat(batch_size, 150, 1)
    Qw = torch.LongTensor(random.sample(wl, 15)).repeat(batch_size, 1)
    Qc = torch.LongTensor(random.sample(cl, 16)).repeat(batch_size, 15, 1)

    p0, p1 = model(Cw, Cc, Qw, Qc)
