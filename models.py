import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from config import config

D = config.connector_dim
Nh = config.num_heads
Dword = config.glove_dim
Dchar = config.char_dim
batch_size = config.batch_size
dropout = config.dropout
dropout_char = config.dropout_char

Dk = D // Nh
Dv = D // Nh
D_cq_att = D * 4
Lc = config.para_limit
Lq = config.ques_limit

def mask_logits(target, mask):
    v = -1e30
    return target + (1 - mask) * v

class PosEncoder(nn.Module):
    def __init__(self, length):
        super().__init__()
        freqs = torch.Tensor(
            [10000 ** (-i / D) if i % 2 == 0 else -10000 ** ((1 - i) / D) for i in range(D)]).unsqueeze(dim=1)
        phases = torch.Tensor([0 if i % 2 == 0 else math.pi / 2 for i in range(D)]).unsqueeze(dim=1)
        pos = torch.arange(length).repeat(D, 1)
        self.pos_encoding = nn.Parameter(torch.sin(torch.add(torch.mul(pos, freqs), phases)), requires_grad=False)

    def forward(self, x):
        x = x + self.pos_encoding
        return x


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
    def __init__(self, layer_num: int, size=D):
        super().__init__()
        self.n = layer_num
        self.linear = nn.ModuleList([nn.Linear(size, size) for _ in range(self.n)])
        self.gate = nn.ModuleList([nn.Linear(size, size) for _ in range(self.n)])

    def forward(self, x):
        x = x.transpose(1, 2)
        for i in range(self.n):
            gate = F.sigmoid(self.gate[i](x))
            nonlinear = F.relu(self.linear[i](x))
            x = gate * nonlinear + (1 - gate) * x
        x = x.transpose(1, 2)
        return x


class SelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        Wo = torch.empty(D, Dv * Nh)
        Wqs = [torch.empty(D, Dk) for _ in range(Nh)]
        Wks = [torch.empty(D, Dk) for _ in range(Nh)]
        Wvs = [torch.empty(D, Dv) for _ in range(Nh)]
        nn.init.kaiming_uniform_(Wo)
        for i in range(Nh):
            nn.init.xavier_uniform_(Wqs[i])
            nn.init.xavier_uniform_(Wks[i])
            nn.init.xavier_uniform_(Wvs[i])
        self.Wo = nn.Parameter(Wo)
        self.Wqs = nn.ParameterList([nn.Parameter(X) for X in Wqs])
        self.Wks = nn.ParameterList([nn.Parameter(X) for X in Wks])
        self.Wvs = nn.ParameterList([nn.Parameter(X) for X in Wvs])

    def forward(self, x, mask):
        WQs, WKs, WVs = [], [], []
        sqrt_dk_inv = 1 / math.sqrt(Dk)
        x = x.transpose(1, 2)
        mask = mask.unsqueeze(2)
        for i in range(Nh):
            WQs.append(torch.matmul(x, self.Wqs[i]))
            WKs.append(torch.matmul(x, self.Wks[i]))
            WVs.append(torch.matmul(x, self.Wvs[i]))
        heads = []
        for i in range(Nh):
            out = torch.bmm(WQs[i], WKs[i].transpose(1, 2))
            out = torch.mul(out, sqrt_dk_inv)
            # not sure... I think `dim` should be 1 since it weighted each column of `WVs[i]`
            out = mask_logits(out, mask)
            out = F.softmax(out, dim=1)
            headi = torch.bmm(out, WVs[i])
            heads.append(headi)
        head = torch.cat(heads, dim=2)
        out = torch.matmul(head, self.Wo)
        return out.transpose(1, 2)


class Embedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d = DepthwiseSeparableConv(Dchar, D, 5, dim=2, bias=True)
        self.conv1d = DepthwiseSeparableConv(Dword + D, D, 5, bias=True)
        self.high = Highway(2)

    def forward(self, ch_emb, wd_emb):
        ch_emb = ch_emb.permute(0, 3, 1, 2)
        ch_emb = F.dropout(ch_emb, p=dropout_char, training=self.training)
        ch_emb = self.conv2d(ch_emb)
        ch_emb = F.relu(ch_emb)
        ch_emb, _ = torch.max(ch_emb, dim=3)
        ch_emb = ch_emb.squeeze()
        wd_emb = F.dropout(wd_emb, p=dropout, training=self.training)
        wd_emb = wd_emb.transpose(1, 2)
        emb = torch.cat([ch_emb, wd_emb], dim=1)
        emb = self.conv1d(emb)
        emb = self.high(emb)
        return emb


class EncoderBlock(nn.Module):
    def __init__(self, conv_num: int, ch_num: int, k: int, length: int):
        super().__init__()
        self.convs = nn.ModuleList([DepthwiseSeparableConv(ch_num, ch_num, k) for _ in range(conv_num)])
        self.self_att = SelfAttention()
        self.fc = nn.Linear(ch_num, ch_num, bias=True)
        self.pos = PosEncoder(length)
        self.normb = nn.LayerNorm([D, length])
        self.norms = nn.ModuleList([nn.LayerNorm([D, length]) for _ in range(conv_num)])
        self.norme = nn.LayerNorm([D, length])
        self.L = conv_num

    def forward(self, x, mask):
        out = self.pos(x)
        res = out
        out = self.normb(out)
        for i, conv in enumerate(self.convs):
            out = conv(out)
            out = F.relu(out)
            out = out + res
            if (i + 1) % 2 == 0:
                p_drop = dropout * (i+1) / self.L
                out = F.dropout(out, p=p_drop, training=self.training)
            res = out
            out = self.norms[i](out)
        out = self.self_att(out, mask)
        out = out + res
        out = F.dropout(out, p=dropout, training=self.training)
        res = out
        out = self.norme(out)
        out = self.fc(out.transpose(1, 2)).transpose(1, 2)
        out = F.relu(out)
        out = out + res
        out = F.dropout(out, p=dropout, training=self.training)
        return out


class CQAttention(nn.Module):
    def __init__(self):
        super().__init__()
        w = torch.empty(D * 3)
        lim = 1/D
        nn.init.uniform_(w, -math.sqrt(lim), math.sqrt(lim))
        self.w = nn.Parameter(w)

    def forward(self, C, Q, cmask, qmask):
        ss = []
        C = C.transpose(1, 2)
        Q = Q.transpose(1, 2)
        cmask = cmask.unsqueeze(2)
        qmask = qmask.unsqueeze(1)
        for i in range(Lq):
            q = Q[:, i, :].unsqueeze(1)
            QCi = torch.mul(q, C)
            Qi = q.expand(batch_size, Lc, D)
            Xi = torch.cat([Qi, C, QCi], dim=2)
            Si = torch.matmul(Xi, self.w).unsqueeze(2)
            ss.append(Si)
        S = torch.cat(ss, dim=2)
        S1 = F.softmax(mask_logits(S, qmask), dim=2)
        S2 = F.softmax(mask_logits(S, cmask), dim=1)
        A = torch.bmm(S1, Q)
        B = torch.bmm(torch.bmm(S1, S2.transpose(1, 2)), C)
        out = torch.cat([C, A, torch.mul(C, A), torch.mul(C, B)], dim=2)
        out = F.dropout(out, p=dropout, training=self.training)
        return out.transpose(1, 2)


class Pointer(nn.Module):
    def __init__(self):
        super().__init__()
        w1 = torch.empty(D * 2)
        w2 = torch.empty(D * 2)
        lim = 3/(2*D)
        nn.init.uniform_(w1, -math.sqrt(lim), math.sqrt(lim))
        nn.init.uniform_(w2, -math.sqrt(lim), math.sqrt(lim))
        self.w1 = nn.Parameter(w1)
        self.w2 = nn.Parameter(w2)

    def forward(self, M1, M2, M3, mask):
        X1 = torch.cat([M1, M2], dim=1)
        X2 = torch.cat([M1, M3], dim=1)
        Y1 = torch.matmul(self.w1, X1)
        Y2 = torch.matmul(self.w2, X2)
        Y1 = mask_logits(Y1, mask)
        Y2 = mask_logits(Y2, mask)
        p1 = F.log_softmax(Y1, dim=1)
        p2 = F.log_softmax(Y2, dim=1)
        return p1, p2


class QANet(nn.Module):
    def __init__(self, word_mat, char_mat):
        super().__init__()
        if config.pretrained_char:
            self.char_emb = nn.Embedding.from_pretrained(torch.Tensor(char_mat))
        else:
            char_mat = torch.Tensor(char_mat)
            # nn.init.kaiming_uniform_(char_mat)
            self.char_emb = nn.Embedding.from_pretrained(char_mat, freeze=False)
        self.word_emb = nn.Embedding.from_pretrained(torch.Tensor(word_mat))
        self.emb = Embedding()
        self.c_emb_enc = EncoderBlock(conv_num=4, ch_num=D, k=7, length=Lc)
        self.q_emb_enc = EncoderBlock(conv_num=4, ch_num=D, k=7, length=Lq)
        self.cq_att = CQAttention()
        self.cq_resizer = DepthwiseSeparableConv(D * 4, D, 5)
        enc_blk = EncoderBlock(conv_num=2, ch_num=D, k=5, length=Lc)
        self.model_enc_blks = nn.ModuleList([enc_blk] * 7)
        self.out = Pointer()

    def forward(self, Cwid, Ccid, Qwid, Qcid):
        cmask = (torch.zeros_like(Cwid) != Cwid).float()
        qmask = (torch.zeros_like(Qwid) != Qwid).float()
        Cw, Cc = self.word_emb(Cwid), self.char_emb(Ccid)
        Qw, Qc = self.word_emb(Qwid), self.char_emb(Qcid)
        C, Q = self.emb(Cc, Cw), self.emb(Qc, Qw)
        Ce = self.c_emb_enc(C, cmask)
        Qe = self.q_emb_enc(Q, qmask)
        X = self.cq_att(Ce, Qe, cmask, qmask)
        M1 = self.cq_resizer(X)
        for enc in self.model_enc_blks: M1 = enc(M1, cmask)
        M2 = M1
        for enc in self.model_enc_blks: M2 = enc(M2, cmask)
        M3 = M2
        for enc in self.model_enc_blks: M3 = enc(M3, cmask)
        p1, p2 = self.out(M1, M2, M3, cmask)
        return p1, p2
