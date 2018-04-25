import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

char_limit = 16
char_dim = 8
char_hidden_size = 100
char_num_layers = 1
char_dir = 2

dropout = 0.2
batch_size = 24
hidden_size = 75
word_emb_size = 300
char_emb_size = char_dir * char_num_layers * char_hidden_size
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
        self.gru = nn.GRU(input_size=in_size, bidirectional=self.bidirectional, num_layers=self.num_layers, hidden_size=self.hidden_size)
        self.h = torch.randn(self.num_layers*self.dir, 1, self.hidden_size)
        self.out_size = self.hidden_size * self.num_layers * self.dir

    def forward(self, input):
        (l, b, in_size) = input.size()
        assert in_size == self.in_size and b == 1
        o, h = self.gru(input, self.h)
        h = h.view(-1)
        return h

# Input is the concatenation of word embedding and its corresponding char embedding
# Output is passage embedding or question embedding
class Encoder(nn.Module):
    def __init__(self, in_size):
        super(Encoder, self).__init__()
        self.bidirectional = True
        self.dir = 2 if self.bidirectional else 1
        self.num_layers = 3
        self.hidden_size = hidden_size
        self.in_size = in_size
        self.gru = nn.GRU(input_size=in_size, bidirectional=self.bidirectional, num_layers=self.num_layers, hidden_size=self.hidden_size)
        self.out_size = self.hidden_size * self.num_layers * self.dir
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, input):
        (l, _, in_size) = input.size()
        hs = torch.zeros(l, self.num_layers * self.dir, batch_size, hidden_size).to(device)
        h = torch.randn(self.num_layers * self.dir, batch_size, self.hidden_size).to(device)
        input = torch.unsqueeze(input, dim=1)
        for i in range(l):
            self.gru.flatten_parameters()
            _, h = self.gru(input[i], h)
            hs[i] = h
        del h, input
        hs_ = hs.permute([0,2,1,3]).contiguous().view(l, batch_size, -1)
        hs = self.dropout(hs_)
        return hs

# Using passage and question to obtain question-aware passage representation
# Co-attention
class PQMatcher(nn.Module):
    def __init__(self, in_size):
        super(PQMatcher, self).__init__()
        self.hidden_size = hidden_size * 2
        self.in_size = in_size
        self.gru = nn.GRUCell(input_size=in_size*2, hidden_size=self.hidden_size)
        self.Wp = nn.Linear(self.in_size*2, self.hidden_size, bias=False)
        self.Wq = nn.Linear(self.in_size*2, self.hidden_size, bias=False)
        self.Wv = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.Wg = nn.Linear(self.in_size*4, self.in_size*4, bias=False)
        self.out_size = self.hidden_size
        self.dropout = nn.Dropout(p=dropout)

    
    def forward(self, up, uq):
        (lp, _, _) = up.size()
        (lq, _, _) = uq.size()
        mixerp, mixerq = torch.arange(lp).long().to(device), torch.arange(lq).long().to(device)
        Up = torch.cat([up, up[mixerp]], dim=2)
        Uq = torch.cat([uq, uq[mixerq]], dim=2)
        vs = torch.zeros(lp, batch_size, self.out_size).to(device)
        v = torch.randn(batch_size, self.hidden_size).to(device)
        V = torch.randn(batch_size, self.hidden_size, 1).to(device)
        
        Uq_ = Uq.permute([1, 0, 2])
        for i in range(lp):
            Wup = self.Wp(Up[i])
            Wuq = self.Wq(Uq)
            Wvv = self.Wv(v)
            x = F.tanh(Wup + Wuq + Wvv).permute([1, 0, 2])
            s = torch.bmm(x, V)
            s = torch.squeeze(s, 2)
            a = F.softmax(s, 1).unsqueeze(1)
            c = torch.bmm(a, Uq_).squeeze()
            r = torch.cat([Up[i], c], dim=1)
            g = F.sigmoid(self.Wg(r))
            r_ = torch.mul(g, r)
            c_ = r_[:, self.in_size*2:]
            v = self.gru(c_, v)
            vs[i] = v
            del Wup, Wuq, Wvv, x, a, s, c, g, r, r_, c_
        del up, uq, Up, Uq, Uq_
        vs = self.dropout(vs)
        return vs

# Input is question-aware passage representation
# Output is self-attention question-aware passage representation
class SelfMatcher(nn.Module):
    def __init__(self, in_size):
        super(SelfMatcher, self).__init__()
        self.hidden_size = in_size
        self.in_size = in_size
        self.gru = nn.GRUCell(input_size=in_size, hidden_size=self.hidden_size)
        self.Wp = nn.Linear(self.in_size, self.hidden_size, bias=False)
        self.Wp_ = nn.Linear(self.in_size, self.hidden_size, bias=False)
        self.out_size = self.hidden_size
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, v):
        (l, _, _) = v.size()
        h = torch.randn(batch_size, self.hidden_size).to(device)
        V = torch.randn(batch_size, self.hidden_size, 1).to(device)
        hs = torch.zeros(l, batch_size, self.out_size).to(device)
        
        for i in range(l):
            Wpv = self.Wp(v[i])
            Wpv_ = self.Wp_(v)
            x = F.tanh(Wpv + Wpv_)
            x = x.permute([1, 0, 2])
            s = torch.bmm(x, V)
            s = torch.squeeze(s, 2)
            a = F.softmax(s, 1).unsqueeze(1)
            c = torch.bmm(a, v.permute([1, 0, 2])).squeeze()
            h = self.gru(c, h)
            hs[i] = h
            logger.gpu_mem_log("SelfMatcher {:002d}".format(i), ['x', 'Wpv', 'Wpv_', 's', 'c', 'hs'], [x.data, Wpv.data, Wpv_.data, s.data, c.data, hs.data])
            del Wpv, Wpv_, x, s, a, c
        hs = self.dropout(hs)
        del h, v
        return hs

# Input is question representation and self-attention question-aware passage representation
# Output are start and end pointer distribution
class Pointer(nn.Module):
    def __init__(self, in_size1, in_size2):
        super(Pointer, self).__init__()
        self.hidden_size = in_size2
        self.in_size1 = in_size1
        self.in_size2 = in_size2
        self.gru = nn.GRUCell(input_size=in_size1, hidden_size=self.hidden_size)
        # Wu uses bias. See formula (11). Maybe Vr is just a bias.
        self.Wu = nn.Linear(self.in_size2, self.hidden_size, bias=True)
        self.Wh = nn.Linear(self.in_size1, self.hidden_size, bias=False)
        self.Wha = nn.Linear(self.in_size2, self.hidden_size, bias=False)
        self.out_size = 1

    def forward(self, h, u):
        (lp, _, _) = h.size()
        (lq, _, _) = u.size()
        v = torch.randn(batch_size, self.hidden_size, 1).to(device)
        u_ = u.permute([1,0,2])
        h_ = h.permute([1,0,2])
        x = F.tanh(self.Wu(u)).permute([1, 0, 2])
        s = torch.bmm(x, v)
        s = torch.squeeze(s, 2)
        a = F.softmax(s, 1).unsqueeze(1)
        r = torch.bmm(a, u_).squeeze()
        x = F.tanh(self.Wh(h)+self.Wha(r)).permute([1, 0, 2])
        s = torch.bmm(x, v)
        s = torch.squeeze(s)
        p1 = F.softmax(s, 1)
        c = torch.bmm(p1.unsqueeze(1), h_).squeeze()
        r = self.gru(c, r)
        x = F.tanh(self.Wh(h) + self.Wha(r)).permute([1, 0, 2])
        s = torch.bmm(x, v)
        s = torch.squeeze(s)
        p2 = F.softmax(s, 1)
        return (p1, p2)


class RNet(nn.Module):
    def __init__(self):
        super(RNet, self).__init__()
        self.encoder = Encoder(emb_size)
        self.pqmatcher = PQMatcher(self.encoder.out_size)
        self.selfmatcher = SelfMatcher(self.pqmatcher.out_size)
        self.pointer = Pointer(self.selfmatcher.out_size, self.encoder.out_size)

    # wemb of P, cemb of P, w of Q, c of Q, Answer
    def forward(self, Pw, Pc, Qw, Qc):
        lp = Pw.size()[0]
        lq = Qw.size()[0]
        P = torch.cat([Pw, Pc], dim=2)
        Q = torch.cat([Qw, Qc], dim=2)
        Up = self.encoder(P)
        Uq = self.encoder(Q)
        v = self.pqmatcher(Up, Uq)
        torch.cuda.empty_cache()
        h = self.selfmatcher(v)
        p1, p2 = self.pointer(h, Uq)
        return p1, p2