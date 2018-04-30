import argparse
import models
from dataset import get_dataset
from dataset import SQuAD
from models import QANet
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.cuda
import torch.backends.cudnn as cudnn
from tqdm import tqdm
import os
import random
import math
import ujson as uj
import evaluation

model_fn = "model.pt"
model_dir = "model/"
log_dir = "log/"
checkpoint = 1000
batch_size = models.batch_size
device = models.device
cudnn.enabled = True
max_char_num = models.max_char_num


def parse_args():
    args = argparse.ArgumentParser(description="An R-net implementation.")
    args.add_argument('--mode', dest='mode', type=str, default='all')
    args.add_argument("--batch_size", dest='batch_size', type=int, default="64")
    args.add_argument("--checkpoint", dest='checkpoint', type=int, default="10000")
    args.add_argument("--epoch", dest='epoch', type=int, default="10")
    return args.parse_args()


def to_batch(pack, data: SQuAD, dataset):
    # tensor representation of passage, question, answer
    # pw is a list of word embeddings.
    # pw[i] == data.word_embedding[dataset.wpassages[pack[0]]]
    assert batch_size == len(pack)
    Cw, Cc, Qw, Qc, As = [], [], [], [], []
    max_cl, max_ql = 0, 0
    for i in range(batch_size):
        pid, qid, aid = pack[i]
        p, q, a = dataset.contexts[pid], dataset.questions[qid], dataset.answers[aid]
        max_cl, max_ql = max(max_cl, len(p)), max(max_ql, len(q))
        Cw.append(p)
        Qw.append(q)
        As.append(a)
    a = torch.zeros(batch_size, 2, device=device).long()
    for i in range(batch_size):
        p, q = Cw[i], Qw[i]
        p_, q_ = [0] * max_cl, [0] * max_ql
        p_[0:len(p)], q_[0:len(q)] = p, q
        Cw[i], Qw[i] = p_, q_
        Cc.append([[0] * 16] * max_cl)
        Qc.append([[0] * 16] * max_ql)
        for j in range(max_cl):
            wid = Cw[i][j]
            if wid == 0: continue
            cs = [data.ctoi[c] if c in data.ctoi else 0 for c in list(data.itow[wid][:max_char_num])]
            Cc[i][j][0:len(cs)] = cs
        for j in range(max_ql):
            wid = Qw[i][j]
            if wid == 0: continue
            cs = [data.ctoi[c] if c in data.ctoi else 0 for c in list(data.itow[wid][:max_char_num])]
            Qc[i][j][0:len(cs)] = cs
        a[i, 0] = As[i][0]
        a[i, 1] = As[i][1]
    Cw = torch.LongTensor(Cw).to(device)
    Cc = torch.LongTensor(Cc).to(device)
    Qw = torch.LongTensor(Qw).to(device)
    Qc = torch.LongTensor(Qc).to(device)
    return Cw, Cc, Qw, Qc, a


def trunk(packs, batch_size):
    bpacks = []
    for i in range(0, len(packs), batch_size):
        bpacks.append(packs[i:i + batch_size])
    ll = len(bpacks[-1])
    if ll < batch_size:
        for j in range(batch_size - ll):
            bpacks[-1].append(random.choice(bpacks[-1]))
    random.shuffle(bpacks)
    return bpacks


def train(epoch, data):
    model = QANet(data).to(device)
    parameters = filter(lambda param: param.requires_grad, model.parameters())
    optimizer = optim.Adam(betas=(0.8, 0.999), eps=1e-7, weight_decay=3e-7, params=parameters)
    crit = 0.001 / math.log2(1000)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda ee: crit * math.log2(
        ee + 1) if ee + 1 <= 1000 else 0.001)
    packs = trunk(data.train.packs, batch_size)
    f_log = open("log/model.log", "w")
    try:
        for ep in range(epoch):
            print("EPOCH {:02d}: ".format(ep))
            l = len(packs)
            for i in tqdm(range(l)):
                pack = packs[i]
                Cw, Cc, Qw, Qc, a = to_batch(pack, data, data.train)
                optimizer.zero_grad()
                out1, out2 = model(Cw, Cc, Qw, Qc)
                loss1 = F.cross_entropy(out1, a[:, 0])
                loss2 = F.cross_entropy(out2, a[:, 1])
                loss = (loss1 + loss2) / 2
                loss.backward()
                scheduler.step()
                if (i+1) % checkpoint == 0:
                    torch.save(model, os.path.join(model_dir, "model-tmp-{:02d}-{}.pt".format(ep, i + 1)))
            em, f1 = test(model, data)
            llog = "EPOCH: {:02d}\tEM: {:6.40f}\tF1: {:6.40f}\n".format(ep + 1, i + 1, em, f1)
            f_log.write(llog)
            f_log.flush()
            random.shuffle(packs)
        torch.save(model, os.path.join(model_dir, model_fn))
    except Exception as e:
        torch.save(model, os.path.join(model_dir, "model-{:02d}-{}.pt".format(ep, i + 1)))
        raise e
    except KeyboardInterrupt as k:
        torch.save(model, os.path.join(model_dir, "model-{:02d}-{}.pt".format(ep, i + 1)))
    return model


def get_anwser(i, j, pid, itow, dataset):
    p = dataset.contexts[pid]
    i, j = min(i, j), max(i, j)
    if j >= len(p): return ""
    ans_ = []
    for t in range(j - i + 1):
        ans_.append(p[i + t])
    ans = ""
    for a in ans_:
        ans += itow[a] + ' '
    return ans[:-1]

def evaluate_from_file(dataset_file, prediction_file):
    with open(dataset_file) as dataset_file:
        dataset_json = uj.load(dataset_file)
        dataset = dataset_json['data']
    with open(prediction_file) as prediction_file:
        predictions = uj.load(prediction_file)
    res = evaluation.evaluate(dataset, predictions)
    return res['exact_match'], res['f1']


def test(model, data):
    packs = trunk(data.dev.packs, batch_size)
    l = len(packs)
    anss = {}
    print("Testing...")
    for i in tqdm(range(l)):
        pack = packs[i]
        Cw, Cc, Qw, Qc, a = to_batch(pack, data, data.dev)
        out1, out2 = model(Cw, Cc, Qw, Qc)
        _, idx1 = torch.max(out1, dim=1)
        _, idx2 = torch.max(out2, dim=1)
        na = torch.cat([idx1.unsqueeze(1), idx2.unsqueeze(1)], dim=1)
        for j in range(batch_size):
            ans = get_anwser(na[j, 0], na[j, 1], pack[j][0], data.itow, data.dev)
            anss[data.dev.question_ids[pack[j][1]]] = ans
    with open('log/answer.json', 'w') as f:
        uj.dump(anss, f)
        f.close()
        em, f1 = evaluate_from_file('tmp/squad/dev-v1.1.json', 'log/answer.json')
        print("EM: {}, F1: {}".format(em, f1))
    return em, f1


def main():
    args = parse_args()
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    data = get_dataset()
    if args.mode == 'all':
        model = train(args.epoch, data)
        test(model, data)
    elif args.mode == 'train':
        train(args.epoch, data)
    elif args.mode == 'test':
        model = torch.load(model_fn)
        test(model, data)
    else:
        print("Wrong arguments!")


if __name__ == '__main__':
    main()