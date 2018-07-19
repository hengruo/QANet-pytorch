import os
import ipdb
import math
import json

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from tqdm import trange
from model import QANet
from config import Config
from Vision import Visualizer
from SQuAD import Train, Dev, Text


class EMA(object):
    def __init__(self, decay):
        self.decay = decay
        self.shadows = {}
        self.devices = {}

    def __len__(self):
        return len(self.shadows)

    def get(self, name: str):
        return self.shadows[name].to(Config.device)

    def set(self, name: str, param: nn.Parameter):
        self.shadows[name] = param.data.to('cpu').clone()
        self.devices[name] = param.data.device

    def update_parameter(self, name: str, param: nn.Parameter):
        if name in self.shadows:
            data = param.data
            new_shadow = self.decay * data + (1.0 - self.decay) * self.get(name)
            param.data.copy_(new_shadow)
            self.shadows[name] = new_shadow.to('cpu').clone()


def training(Train, vis):
    Acc = 0
    Total = 0
    Loss = []

    # C = torch.load('./data/char2vector.pt')
    C=torch.randn(Config.vocab,Config.char_embd)
    W = Text.vocab.vectors
    # W[0]=torch.randn(Config.word_embd)
    model = QANet(W, C).to(Config.device)
    # ema = EMA(Config.ema_decay)
    # for name, p in model.named_parameters():
    #     if p.requires_grad:
    #         ema.set(name, p)
    parameters = filter(lambda param: param.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(lr=Config.base_lr, betas=(Config.beta1, Config.beta2),
                                 eps=1e-7, weight_decay=3e-7, params=parameters)
    cr = Config.lr / math.log2(Config.warm_num)
    lamb = lambda ee: cr * math.log2(ee + 1) if ee < Config.warm_num else Config.lr
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lamb)

    LR = []
    for i in trange(Config.check_point * Config.step,ncols=50):

        # Debug
        if (os.path.exists(Config.debug)):
            ipdb.set_trace()

        model.train()
        batch = next(iter(Train))
        batch.question = batch.question[:, :Config.question_limit]

        # Cuda
        batch.question = batch.question.to(Config.device)
        batch.context = batch.context.to(Config.device)
        batch.start = batch.start.to(Config.device)
        batch.end = batch.end.to(Config.device)

        # Caculate
        ps, pe = model(batch.context, batch.question)
        ls = F.nll_loss(ps, batch.start, size_average=True)
        le = F.nll_loss(pe, batch.end, size_average=True)
        loss = (ls + le) / 2
        Loss.append(loss.item())
        vis.plot('LOSS', loss.item())
        # Record
        Total += Config.batch
        s = torch.argmax(ps, 1)
        e = torch.argmax(pe, 1)
        Acc += torch.sum((s == batch.start) & (e == batch.end)).item()

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # for name, p in model.named_parameters():
        #     if (p.requires_grad):
        #         ema.update_parameter(name, p)
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.grad_clip)

        if (i % Config.check_point == 0):
            
            # vis.plot('Compare',loss.item())
            vis.plot('Acc', Acc / Total)
            j = i // Config.check_point
            print("Step{}:{}/{},EM:{}%".format(j, Acc, Total, Acc / Total))
            if (i == 1000):
                step = np.array(range(len(LR)))
                LR = np.array(LR)
                vis.draw(step, LR)
            Acc = 0
            Total = 0
            Loss = []


vis = Visualizer(Config.name)
training(Train, vis)
