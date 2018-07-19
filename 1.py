import ipdb
import math
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Config


def mask_logits(target, mask):
    return target * mask + (1 - mask) * (-1e30)


def word2char(word):
    with open('./data/word2char.json', 'r') as f:
        w2c = json.load(f)
    word = word.tolist()
    char = []
    for w in word:
        tmp = []
        for ww in w:
            tmp.append(w2c[str(ww)])
        char.append(tmp)

    char = torch.tensor(char, dtype=torch.long).to(Config.device)
    return char


class Position(nn.Module):
    def __init__(self, length):
        super(Position, self).__init__()
        freqs = torch.Tensor(
            [10000 ** (-i / Config.dimension) if i % 2 == 0 else -10000 ** ((1 - i) / Config.dimension) for i in range(Config.dimension)])
        freqs = freqs.unsqueeze(dim=1)
        phases = torch.Tensor([0 if i % 2 == 0 else math.pi / 2 for i in range(Config.dimension)]).unsqueeze(dim=1)
        pos = torch.arange(length).repeat(Config.dimension, 1)
        self.position = nn.Parameter(torch.sin(torch.add(torch.mul(pos, freqs), phases)), requires_grad=False)

    def forward(self, x):
        x = x + self.position
        return x


class DepthConv(nn.Module):
    def __init__(self, in_ch, out_ch, k, dim=1, bias=False):
        super(DepthConv, self).__init__()
        if (dim == 1):
            self.depth = nn.Conv1d(in_channels=in_ch, out_channels=in_ch, kernel_size=k,
                                   groups=in_ch, padding=k // 2, bias=bias)
            self.point = nn.Conv1d(in_channels=in_ch, out_channels=out_ch, kernel_size=1,
                                   padding=0, bias=bias)
        elif (dim == 2):
            self.depth = nn.Conv2d(in_channels=in_ch, out_channels=in_ch, kernel_size=k,
                                   groups=in_ch, padding=k // 2, bias=bias)
            self.point = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=1,
                                   padding=0, bias=bias)
        else:
            raise Exception("Wrong dimension for DeepConv!")

    def forward(self, x):
        x = self.point(self.depth(x))
        return x


class HighWay(nn.Module):
    def __init__(self, layer_num, size=Config.word_embd + Config.char_embd):
        super(HighWay, self).__init__()
        self.n = layer_num
        self.linear = nn.ModuleList([nn.Linear(size, size) for _ in range(self.n)])
        self.gate = nn.ModuleList([nn.Linear(size, size) for _ in range(self.n)])

    def forward(self, x):
        '''
        ------------------------
        Note:
            B for batch
            N for sequence length
            D for dimension
            W for word embedding
            C for char embedding
        -------------------------

        '''
        x = x.transpose(1, 2)  # (B,N,W+C)
        for i in range(self.n):
            t = F.sigmoid(self.gate[i](x))  # (B,N,W+C)
            h = F.relu(self.linear[i](x))  # (B,N,W+C)
            x = t * h + (1 - t) * x
        x = x.transpose(1, 2)  # (B,W+C,N)
        return x


class SelfAttention(nn.Module):
    def __init__(self):
        super(SelfAttention, self).__init__()
        self.Da = Config.dimension // Config.head_num
        Wo = torch.empty(Config.dimension, self.Da * Config.head_num)
        Wq = [torch.empty(Config.dimension, self.Da) for _ in range(Config.head_num)]
        Wk = [torch.empty(Config.dimension, self.Da) for _ in range(Config.head_num)]
        Wv = [torch.empty(Config.dimension, self.Da) for _ in range(Config.head_num)]
        nn.init.kaiming_uniform_(Wo)

        for i in range(Config.head_num):
            nn.init.xavier_uniform_(Wq[i])
            nn.init.xavier_uniform_(Wk[i])
            nn.init.xavier_uniform_(Wv[i])

        self.Wo = nn.Parameter(Wo)
        self.Wq = nn.ParameterList([nn.Parameter(X) for X in Wq])
        self.Wk = nn.ParameterList([nn.Parameter(X) for X in Wk])
        self.Wv = nn.ParameterList([nn.Parameter(X) for X in Wv])

    def forward(self, x, mask):
        '''
        ------------------------
        Note:
            B for batch
            N for sequence length
            D for dimension
            Da for dimension/head
        -------------------------
        '''

        WQ, WK, WV = [], [], []
        Dk = 1 / math.sqrt(self.Da)
        x = x.transpose(1, 2)  # (B,N,D)

        hmask = mask.unsqueeze(1)  # (B,1,N)
        vmask = mask.unsqueeze(2)  # (B,N,1)

        for i in range(Config.head_num):
            WQ.append(torch.matmul(x, self.Wq[i]))  # (B,N,Da)
            WK.append(torch.matmul(x, self.Wk[i]))  # (B,N,Da)
            WV.append(torch.matmul(x, self.Wv[i]))  # (B,N,Da)

        heads = []
        for i in range(Config.head_num):
            out = torch.bmm(WQ[i], WK[i].transpose(1, 2))  # (B,N,N)
            out = torch.mul(out, Dk)  # (B,N,N)
            out = mask_logits(out, hmask)  # (B,N,N)
            out = F.softmax(out, dim=2) * vmask  # (B,N,N)
            h = torch.bmm(out, WV[i])  # (B,N,Da)
            heads.append(h)

        head = torch.cat(heads, dim=2)  # (B,N,D)
        out = torch.matmul(head, self.Wo)  # (B,N,D)
        out = out.transpose(1, 2)  # (B,D,N)
        return out


class Embedding(nn.Module):
    def __init__(self):
        super(Embedding, self).__init__()
        self.conv = DepthConv(Config.char_embd, Config.char_embd, 5, dim=2, bias=True)

        self.high = HighWay(2)

    def forward(self, char, word):
        '''
        ------------------------
        Note:
            B for batch
            N for sequence length
            M for word length
            W for word embedding
            C for char embedding
            D for dimension
            DA for dimension/head
        -------------------------

        '''

        # Char Embedding
        char = char.permute(0, 3, 1, 2)  # (B,C,N,M)
        char = F.dropout(char, p=Config.dropout_char, training=self.training)
        char = self.conv(char)  # (B,C,N,M)
        char = F.relu(char)
        char, _ = torch.max(char, dim=3)  # (B,C,N,1)
        char = char.squeeze()  # (B,C,N)

        # Word Embedding
        word = F.dropout(word, p=Config.dropout, training=self.training)
        word = word.transpose(1, 2)  # (B,W,N)

        # Concatenate
        embd = torch.cat([char, word], dim=1)  # (B,W+C,N)
        embd = self.high(embd)  # (B,W+C,N)

        return embd


class Encoder(nn.Module):
    def __init__(self, conv_num, ch_num, k, length):
        super(Encoder, self).__init__()
        self.convs = nn.ModuleList([DepthConv(ch_num, ch_num, k) for _ in range(conv_num)])
        self.attention = SelfAttention()
        self.fc = nn.Linear(ch_num, ch_num, bias=True)
        self.pos = Position(length)
        self.normb = nn.LayerNorm([Config.dimension, length])
        self.norms = nn.ModuleList([nn.LayerNorm([Config.dimension, length]) for _ in range(conv_num)])
        self.norme = nn.LayerNorm([Config.dimension, length])
        self.L = conv_num

    def forward(self, x, mask):
        '''
        ------------------------
        Note:
            B for batch
            N for sequence length
            D for dimension
        -------------------------

        '''
        out = self.pos(x)  # (B,D,N)
        res = out
        out = self.normb(out)  # (B,D,N)
        for i, conv in enumerate(self.convs):
            out = conv(out)  # (B,D,N)
            out = F.relu(out)
            out = out + res
            if ((i + 1) % 2 == 0):
                drop = Config.dropout * (i + 1) / self.L
                out = F.dropout(out, p=drop, training=self.training)
            res = out
            out = self.norms[i](out)

        out = self.attention(out, mask)  # (B,D,N)
        out = out + res
        out = F.dropout(out, Config.dropout, training=self.training)
        res = out
        out = self.norme(out)
        out = self.fc(out.transpose(1, 2)).transpose(1, 2)  # (B,D,N)
        out = F.relu(out)
        out = out + res
        out = F.dropout(out, Config.dropout, training=self.training)  # (B,D,N)
        return out


class CQAttention(nn.Module):
    def __init__(self):
        super(CQAttention, self).__init__()
        w = torch.empty(Config.dimension * 3)
        lim = 1 / Config.dimension
        nn.init.uniform_(w, -math.sqrt(lim), math.sqrt(lim))
        self.w = nn.Parameter(w)

    def forward(self, C, Q, cmask, qmask):
        '''
        ------------------------
        Note:
            B for batch
            N for sequence length(c/q)
            D for dimension
            Da for dimension/head
            W for word embedding
            C for char embedding
            M for word length
        -------------------------
        '''
        C = C.transpose(1, 2)  # (B,Nc,D)
        Q = Q.transpose(1, 2)  # (B,Nq,D)

        cmask = cmask.unsqueeze(2)  # (B,Nc,1)
        qmask = qmask.unsqueeze(1)  # (B,1,Nq)

        shape = (C.size(0), Config.context_limit, Config.question_limit, Config.dimension)
        Ct = C.unsqueeze(2).expand(shape)  # (B,Nc,Nq,D)
        Qt = Q.unsqueeze(1).expand(shape)  # (B,Nc,Nq,D)
        CQ = torch.mul(Ct, Qt)  # (B,Nc,Nq,D)
        S = torch.cat([Ct, Qt, CQ], dim=3)  # (B,Nc,Nq,3*D)
        S = torch.matmul(S, self.w)  # (B,Nc,Nq)
        Sq = F.softmax(mask_logits(S, qmask), dim=2)  # (B,Nc,Nq)
        Sc = F.softmax(mask_logits(S, cmask), dim=1)  # (B,Nc,Nq)
        A = torch.bmm(Sq, Q)  # (B,Nc,D)
        B = torch.bmm(torch.bmm(Sq, Sc.transpose(1, 2)), C)  # (B,Nc,D)
        out = torch.cat([C, A, torch.mul(C, A), torch.mul(C, B)], dim=2)  # (B,Nc,4*D)
        out = F.dropout(out, p=Config.dropout, training=self.training)
        out = out.transpose(1, 2)  # (B,4*D,Nc)
        return out


class Pointer(nn.Module):
    def __init__(self):
        super().__init__()
        w1 = torch.empty(Config.dimension * 2)
        w2 = torch.empty(Config.dimension * 2)
        lim = 3 / (2 * Config.dimension)
        nn.init.uniform_(w1, -math.sqrt(lim), math.sqrt(lim))
        nn.init.uniform_(w2, -math.sqrt(lim), math.sqrt(lim))
        self.w1 = nn.Parameter(w1)
        self.w2 = nn.Parameter(w2)

    def forward(self, M1, M2, M3, mask):
        X1 = torch.cat([M1, M2], dim=1)  # (B,2*D,N)
        X2 = torch.cat([M1, M3], dim=1)  # (B,2*D,N)
        Y1 = torch.matmul(self.w1, X1)  # (B,N)
        Y2 = torch.matmul(self.w2, X2)  # (B,N)
        Y1 = mask_logits(Y1, mask)  # (B,N)
        Y2 = mask_logits(Y2, mask)  # (B,N)
        p1 = F.log_softmax(Y1, dim=1)  # (B,N)
        p2 = F.log_softmax(Y2, dim=1)  # (B,N)
        return p1, p2


class QANet(nn.Module):
    def __init__(self, word, char):
        super(QANet, self).__init__()

        self.word_embd = nn.Embedding.from_pretrained(word)
        self.char_embd = nn.Embedding.from_pretrained(char, freeze=False)
        self.embd = Embedding()
        self.context_conv = DepthConv(Config.char_embd + Config.word_embd, Config.dimension, 5)
        self.question_conv = DepthConv(Config.char_embd + Config.word_embd, Config.dimension, 5)
        self.context_encoder = Encoder(conv_num=4, ch_num=Config.dimension, k=7, length=Config.context_limit)
        self.question_encoder = Encoder(conv_num=4, ch_num=Config.dimension, k=7, length=Config.question_limit)
        self.cq_attention = CQAttention()
        self.cq_resizer = DepthConv(Config.dimension * 4, Config.dimension, 5)
        enc = Encoder(conv_num=2, ch_num=Config.dimension, k=5, length=Config.context_limit)
        self.encoder = nn.ModuleList([enc] * 7)
        self.out = Pointer()

    def forward(self, context, question):
        '''
        ------------------------
        Note:
            B for batch
            N for sequence length(c/q)
            D for dimension
            Da for dimension/head
            W for word embedding
            C for char embedding
            M for word length
        -------------------------
        '''
        cmask = (torch.ones_like(context) != context).float()  # (B,Nc)
        qmask = (torch.ones_like(question) != question).float()  # (B,Nq)

        con_word = self.word_embd(context)  # (B,Nc,W)
        que_word = self.word_embd(question)  # (B,Nq,W)

        con_char = self.char_embd(word2char(context))  # (B,Nc,M,C)
        que_char = self.char_embd(word2char(question))  # (B,Nq,M,C)

        C = self.embd(con_char, con_word)  # (B,W+C,Nc)
        Q = self.embd(que_char, que_word)  # (B,W+C,Nq)
        C = self.context_conv(C)  # (B,D,Nc)
        Q = self.question_conv(Q)  # (B,D,Nq)
        C = self.context_encoder(C, cmask)  # (B,D,Nc)
        Q = self.question_encoder(Q, qmask)  # (B,D,Nq)
        X = self.cq_attention(C, Q, cmask, qmask)  # (B,4*D,Nc)

        M1 = self.cq_resizer(X)  # (B,D,Nc)
        for enc in self.encoder:
            M1 = enc(M1, cmask)
        M2 = M1  # (B,D,Nc)
        for enc in self.encoder:
            M2 = enc(M2, cmask)
        M3 = M2  # (B,D,Nc)
        for enc in self.encoder:
            M3 = enc(M3, cmask)

        ps, pe = self.out(M1, M2, M3, cmask)  # (B,N)
        return ps, pe
