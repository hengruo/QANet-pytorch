import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import config

d_model = config.d_model
n_head = config.num_heads
d_word = config.glove_dim
d_char = config.char_dim
batch_size = config.batch_size
dropout = config.dropout
dropout_char = config.dropout_char

d_k = d_model // n_head
d_cq = d_model * 4
len_c = config.para_limit
len_q = config.ques_limit


def mask_logits(target, mask):
    return target * (1-mask) + mask * (-1e30)


class PosEncoder(nn.Module):
    def __init__(self, length):
        super().__init__()
        freqs = torch.Tensor(
            [10000 ** (-i / d_model) if i % 2 == 0 else -10000 ** ((1 - i) / d_model) for i in range(d_model)]).unsqueeze(dim=1)
        phases = torch.Tensor([0 if i % 2 == 0 else math.pi / 2 for i in range(d_model)]).unsqueeze(dim=1)
        pos = torch.arange(length).repeat(d_model, 1).to(torch.float)
        self.pos_encoding = nn.Parameter(torch.sin(torch.add(torch.mul(pos, freqs), phases)), requires_grad=False)

    def forward(self, x):
        x = x + self.pos_encoding
        return x


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch, k, dim=1, bias=True):
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
        nn.init.kaiming_normal_(self.depthwise_conv.weight)
        nn.init.constant_(self.depthwise_conv.bias, 0.0)
        nn.init.kaiming_normal_(self.depthwise_conv.weight)
        nn.init.constant_(self.pointwise_conv.bias, 0.0)

    def forward(self, x):
        return self.pointwise_conv(self.depthwise_conv(x))


class Highway(nn.Module):
    def __init__(self, layer_num: int, size: int):
        super().__init__()
        self.n = layer_num
        self.linear = nn.ModuleList([nn.Linear(size, size) for _ in range(self.n)])
        self.gate = nn.ModuleList([nn.Linear(size, size) for _ in range(self.n)])

    def forward(self, x):
        x = x.transpose(1, 2)
        for i in range(self.n):
            gate = torch.sigmoid(self.gate[i](x))
            nonlinear = F.relu(self.linear[i](x))
            x = gate * nonlinear + (1 - gate) * x
        x = x.transpose(1, 2)
        return x


class SelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        Wo = torch.empty(d_model, d_k * n_head)
        Wqs = [torch.empty(d_model, d_k) for _ in range(n_head)]
        Wks = [torch.empty(d_model, d_k) for _ in range(n_head)]
        Wvs = [torch.empty(d_model, d_k) for _ in range(n_head)]
        nn.init.kaiming_uniform_(Wo)
        for i in range(n_head):
            nn.init.xavier_uniform_(Wqs[i])
            nn.init.xavier_uniform_(Wks[i])
            nn.init.xavier_uniform_(Wvs[i])
        self.Wo = nn.Parameter(Wo)
        self.Wqs = nn.ParameterList([nn.Parameter(X) for X in Wqs])
        self.Wks = nn.ParameterList([nn.Parameter(X) for X in Wks])
        self.Wvs = nn.ParameterList([nn.Parameter(X) for X in Wvs])

    def forward(self, x, mask):
        WQs, WKs, WVs = [], [], []
        sqrt_d_k_inv = 1 / math.sqrt(d_k)
        x = x.transpose(1, 2)
        hmask = mask.unsqueeze(1)
        vmask = mask.unsqueeze(2)
        for i in range(n_head):
            WQs.append(torch.matmul(x, self.Wqs[i]))
            WKs.append(torch.matmul(x, self.Wks[i]))
            WVs.append(torch.matmul(x, self.Wvs[i]))
        heads = []
        for i in range(n_head):
            out = torch.bmm(WQs[i], WKs[i].transpose(1, 2))
            out = torch.mul(out, sqrt_d_k_inv)
            # not sure... I think `dim` should be 2 since it weighted each column of `WVs[i]`
            out = mask_logits(out, hmask)
            out = F.softmax(out, dim=2) * vmask
            headi = torch.bmm(out, WVs[i])
            heads.append(headi)
        head = torch.cat(heads, dim=2)
        out = torch.matmul(head, self.Wo)
        return out.transpose(1, 2)

class MultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, d_model)
        self.a = 1 / math.sqrt(d_k)

    def forward(self, x, mask):
        bs, _, l_x = x.size()
        x = x.transpose(1,2)
        k = self.k_linear(x).view(bs, l_x, n_head, d_k)
        q = self.q_linear(x).view(bs, l_x, n_head, d_k)
        v = self.v_linear(x).view(bs, l_x, n_head, d_k)
        q = q.permute(2, 0, 1, 3).contiguous().view(bs*n_head, l_x, d_k)
        k = k.permute(2, 0, 1, 3).contiguous().view(bs*n_head, l_x, d_k)
        v = v.permute(2, 0, 1, 3).contiguous().view(bs*n_head, l_x, d_k)
        mask = mask.unsqueeze(1).expand(-1, l_x, -1).repeat(n_head, 1, 1)
        
        attn = torch.bmm(q, k.transpose(1, 2)) * self.a
        attn = mask_logits(attn, mask)
        attn = F.softmax(attn, dim=2)
        attn = self.dropout(attn)
            
        out = torch.bmm(attn, v)
        out = out.view(n_head, bs, l_x, d_k).permute(1,2,0,3).contiguous().view(bs, l_x, d_model)
        out = self.fc(out)
        out = self.dropout(out)
        return out.transpose(1,2)

class Embedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d = DepthwiseSeparableConv(d_char, d_char, 5, dim=2)
        self.high = Highway(2, d_word+d_char)

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
        emb = self.high(emb)
        return emb


class EncoderBlock(nn.Module):
    def __init__(self, conv_num: int, ch_num: int, k: int, length: int):
        super().__init__()
        self.convs = nn.ModuleList([DepthwiseSeparableConv(ch_num, ch_num, k) for _ in range(conv_num)])
        self.self_att = MultiHeadAttention()
        self.fc = nn.Linear(ch_num, ch_num, bias=True)
        self.pos = PosEncoder(length)
        # self.norm = nn.LayerNorm([d_model, length])
        self.normb = nn.LayerNorm([d_model, length])
        self.norms = nn.ModuleList([nn.LayerNorm([d_model, length]) for _ in range(conv_num)])
        self.norme = nn.LayerNorm([d_model, length])
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
                p_drop = dropout * (i + 1) / self.L
                out = F.dropout(out, p=p_drop, training=self.training)
            res = out
            out = self.norms[i](out)
        # print("Before attention: {}".format(out.size()))
        out = self.self_att(out, mask)
        # print("After attention: {}".format(out.size()))
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
        w = torch.empty(d_model * 3)
        lim = 1 / d_model
        nn.init.uniform_(w, -math.sqrt(lim), math.sqrt(lim))
        self.w = nn.Parameter(w)

    def forward(self, C, Q, cmask, qmask):
        ss = []
        C = C.transpose(1, 2)
        Q = Q.transpose(1, 2)
        cmask = cmask.unsqueeze(2)
        qmask = qmask.unsqueeze(1)
        
        shape = (C.size(0), C.size(1), Q.size(1), C.size(2))
        Ct = C.unsqueeze(2).expand(shape)
        Qt = Q.unsqueeze(1).expand(shape)
        CQ = torch.mul(Ct, Qt)
        S = torch.cat([Ct, Qt, CQ], dim=3)
        S = torch.matmul(S, self.w)
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
        w1 = torch.empty(d_model * 2)
        w2 = torch.empty(d_model * 2)
        lim = 3 / (2 * d_model)
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
        self.char_emb = nn.Embedding.from_pretrained(torch.Tensor(char_mat), freeze=config.pretrained_char)
        self.word_emb = nn.Embedding.from_pretrained(torch.Tensor(word_mat))
        self.emb = Embedding()
        self.context_conv = DepthwiseSeparableConv(d_word+d_char,d_model, 5)
        self.question_conv = DepthwiseSeparableConv(d_word+d_char,d_model, 5)
        self.c_emb_enc = EncoderBlock(conv_num=4, ch_num=d_model, k=7, length=len_c)
        self.q_emb_enc = EncoderBlock(conv_num=4, ch_num=d_model, k=7, length=len_q)
        self.cq_att = CQAttention()
        self.cq_resizer = DepthwiseSeparableConv(d_model * 4, d_model, 5)
        enc_blk = EncoderBlock(conv_num=2, ch_num=d_model, k=5, length=len_c)
        self.model_enc_blks = nn.ModuleList([enc_blk] * 7)
        self.out = Pointer()

    def forward(self, Cwid, Ccid, Qwid, Qcid):
        cmask = (torch.zeros_like(Cwid) == Cwid).float()
        qmask = (torch.zeros_like(Qwid) == Qwid).float()
        Cw, Cc = self.word_emb(Cwid), self.char_emb(Ccid)
        Qw, Qc = self.word_emb(Qwid), self.char_emb(Qcid)
        C, Q = self.emb(Cc, Cw), self.emb(Qc, Qw)
        C = self.context_conv(C)  
        Q = self.question_conv(Q)  
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