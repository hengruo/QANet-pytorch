import torch
import torch.nn as nn
import torch.nn.functional as F
import math

dropout = 0.1
dropout_w = 0.1
dropout_c = 0.05
batch_size = 24
d_model = 128
h = 8
d_k = d_model // h
d_v = d_model // h
cq_att_size = d_model * 4
word_emb_size = 300
char_emb_size = 200
emb_size = word_emb_size + char_emb_size
training = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

freqs = {}
phases = {}

def get_freq(d: int):
    return torch.Tensor(
        [10000 ** (-i / d) if i % 2 == 0 else -10000 ** (-(i - 1) / d) for i in range(d)]).unsqueeze(dim=1).to(device)


def get_phase(d: int):
    return torch.Tensor([0 if i % 2 == 0 else math.pi / 2 for i in range(d)]).unsqueeze(dim=1).to(device)

def norm(x, eps=1e-6):
    mean = x.mean(-1, keepdim=True)
    std = x.std(-1, keepdim=True)
    return (x - mean) / (std + eps)

def pos_encoding(x):
    (_, d, l) = x.size()
    pos = torch.arange(l).repeat(d, 1).to(device)
    tmp1 = torch.mul(pos, freqs[d])
    tmp2 = torch.add(tmp1, phases[d])
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
    def __init__(self, in_ch, out_ch, k):
        super().__init__()
        self.depthwise_conv = nn.Conv1d(in_channels=in_ch, out_channels=in_ch, kernel_size=k, groups=in_ch,
                                        padding=k//2)
        self.pointwise_conv = nn.Conv1d(in_channels=in_ch, out_channels=out_ch, kernel_size=1, padding=0)

    def forward(self, x):
        return self.pointwise_conv(self.depthwise_conv(x))


class SelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.Wqs = [torch.randn(batch_size, d_k, d_model, device=device, requires_grad=True) for _ in range(h)]
        self.Wks = [torch.randn(batch_size, d_k, d_model, device=device, requires_grad=True) for _ in range(h)]
        self.Wvs = [torch.randn(batch_size, d_v, d_model, device=device, requires_grad=True) for _ in range(h)]
        self.Wo = torch.randn(batch_size, d_v * h, d_model, device=device, requires_grad=True)

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

class ResizingConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int):
        super().__init__()
        self.conv = DepthwiseSeparableConv(in_ch, out_ch, k)

    def forward(self, x):
        return self.conv(x)

class EncoderBlock(nn.Module):
    def __init__(self, conv_num: int, in_ch: int, out_ch: int, k: int):
        super().__init__()
        if in_ch not in freqs:
            freqs[in_ch] = get_freq(in_ch)
            phases[in_ch] = get_phase(in_ch)
        self.convs = nn.ModuleList([DepthwiseSeparableConv(out_ch, out_ch, k) for _ in range(conv_num)])
        self.self_att = SelfAttention()
        self.W = torch.randn(batch_size, out_ch, out_ch, device=device, requires_grad=True)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = pos_encoding(x)
        res = out
        for conv in self.convs:
            out = norm(out)
            out = conv(out)
            out = self.relu(out)
            out = res + out
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
        self.W = torch.randn(batch_size, 1, d_model * 3, device=device, requires_grad=True)

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
        self.W0 = torch.randn(batch_size, 1, d_model * 2, device=device, requires_grad=True)
        self.W1 = torch.randn(batch_size, 1, d_model * 2, device=device, requires_grad=True)

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
        self.c_emb_rez = ResizingConv(in_ch=emb_size, out_ch=d_model, k=7)
        self.q_emb_rez = ResizingConv(in_ch=emb_size, out_ch=d_model, k=7)
        self.c_emb_enc = EncoderBlock(conv_num=4, in_ch=d_model, out_ch=d_model, k=7)
        self.q_emb_enc = EncoderBlock(conv_num=4, in_ch=d_model, out_ch=d_model, k=7)
        self.cq_att = CQAttention()
        self.cq_rez = ResizingConv(in_ch=d_model*4, out_ch=d_model, k=5)
        self.model_enc0 = EncoderBlock(conv_num=2, in_ch=d_model, out_ch=d_model, k=5)
        self.model_enc1 = EncoderBlock(conv_num=2, in_ch=d_model, out_ch=d_model, k=5)
        self.model_enc2 = EncoderBlock(conv_num=2, in_ch=d_model, out_ch=d_model, k=5)
        self.model_enc3 = EncoderBlock(conv_num=2, in_ch=d_model, out_ch=d_model, k=5)
        self.out = Pointer()

    def forward(self, Cwid, Ccid, Qwid, Qcid):
        Cw, Cc = self.word_emb(Cwid), self.char_emb(Ccid)
        Qw, Qc = self.word_emb(Qwid), self.char_emb(Qcid)
        Cc, _ = torch.max(Cc, dim=2)
        Qc, _ = torch.max(Qc, dim=2)
        Cw, Qw = F.dropout(Cw, dropout_w, training=training), F.dropout(Qw, dropout_w, training=training)
        Cc, Qc = F.dropout(Cc, dropout_w, training=training), F.dropout(Qc, dropout_w, training=training)
        C, Q = torch.cat([Cw, Cc], dim=2).transpose(1,2), torch.cat([Qw, Qc], dim=2).transpose(1,2)
        C = self.c_emb_rez(C)
        Q = self.q_emb_rez(Q)
        C = self.c_emb_enc(C)
        Q = self.q_emb_enc(Q)
        X = self.cq_att(C, Q)
        X = self.cq_rez(X)
        M0 = self.model_enc0(self.model_enc1(self.model_enc1(X)))
        M1 = self.model_enc1(self.model_enc2(self.model_enc2(M0)))
        M2 = self.model_enc2(self.model_enc3(self.model_enc3(M1)))
        p0, p1 = self.out(M0, M1, M2)
        p0, p1 = torch.exp(p0), torch.exp(p1)
        return p0, p1



if __name__ == "__main__":
    model = QANet()
    C = torch.randn(batch_size, emb_size, 200)
    Q = torch.randn(batch_size, emb_size, 20)
    p0, p1 = model(C, Q)
