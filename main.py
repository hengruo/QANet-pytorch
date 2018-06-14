from config import config, device
from preproc import preproc
from absl import app
import math
import os
import numpy as np
import ujson as json
import re
from collections import Counter
import string
from tqdm import tqdm
import random
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.cuda
from torch.utils.data import Dataset

'''
Some functions are from the official evaluation script.
'''

class SQuADDataset(Dataset):
    def __init__(self, npz_file, num_steps, batch_size):
        data = np.load(npz_file)
        self.context_idxs = torch.from_numpy(data["context_idxs"]).long()
        self.context_char_idxs = torch.from_numpy(data["context_char_idxs"]).long()
        self.ques_idxs = torch.from_numpy(data["ques_idxs"]).long()
        self.ques_char_idxs = torch.from_numpy(data["ques_char_idxs"]).long()
        self.y1s = torch.from_numpy(data["y1s"]).long()
        self.y2s = torch.from_numpy(data["y2s"]).long()
        self.ids = torch.from_numpy(data["ids"]).long()
        num = len(self.ids)
        self.num_steps = num_steps
        self.batch_size = batch_size
        idxs = list(range(num))
        self.idx_map = []
        i, j = 0, num
        num_items = num_steps * batch_size
        while j <= num_items:
            random.shuffle(idxs)
            self.idx_map += idxs.copy()
            i = j
            j += num
        random.shuffle(idxs)
        self.idx_map += idxs[:num_items - i]

    def __len__(self):
        return self.num_steps

    def __getitem__(self, item):
        idxs = torch.LongTensor(self.idx_map[item:item + self.batch_size])
        res = (self.context_idxs[idxs], self.context_char_idxs[idxs], self.ques_idxs[idxs], self.ques_char_idxs[idxs],
               self.y1s[idxs],
               self.y2s[idxs], self.ids[idxs])
        return res


def convert_tokens(eval_file, qa_id, pp1, pp2):
    answer_dict = {}
    remapped_dict = {}
    for qid, p1, p2 in zip(qa_id, pp1, pp2):
        context = eval_file[str(qid)]["context"]
        spans = eval_file[str(qid)]["spans"]
        uuid = eval_file[str(qid)]["uuid"]
        start_idx = spans[p1][0]
        end_idx = spans[p2][1]
        answer_dict[str(qid)] = context[start_idx: end_idx]
        remapped_dict[uuid] = context[start_idx: end_idx]
    return answer_dict, remapped_dict


def evaluate(eval_file, answer_dict):
    f1 = exact_match = total = 0
    for key, value in answer_dict.items():
        total += 1
        ground_truths = eval_file[key]["answers"]
        prediction = value
        exact_match += metric_max_over_ground_truths(
            exact_match_score, prediction, ground_truths)
        f1 += metric_max_over_ground_truths(f1_score,
                                            prediction, ground_truths)
    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total
    return {'exact_match': exact_match, 'f1': f1}


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def train(model, optimizer, dataset, epoch, length):
    model.train()
    losses = []
    for i in tqdm(range(1, length + 1), total=length):
        model.zero_grad()
        (Cwid, Ccid, Qwid, Qcid, y1, y2, ids) = dataset[i]
        Cwid, Ccid, Qwid, Qcid = Cwid.to(device), Ccid.to(device), Qwid.to(device), Qcid.to(device)
        p1, p2 = model(Cwid, Ccid, Qwid, Qcid)
        y1, y2 = y1.to(device), y2.to(device)
        loss1 = F.cross_entropy(p1, y1)
        loss2 = F.cross_entropy(p2, y2)
        loss = loss1 + loss2
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
    loss_avg = np.mean(losses)
    print("EPOCH {:8d} loss {:8f}\n".format(epoch, loss_avg))


def test(model, dataset, eval_file, epoch):
    model.eval()
    answer_dict = {}
    losses = []
    num_batches = len(dataset)
    with torch.no_grad():
        for i in tqdm(range(1, num_batches + 1), total=num_batches):
            Cwid, Ccid, Qwid, Qcid, y1, y2, ids = dataset[i]
            Cwid, Ccid, Qwid, Qcid = Cwid.to(device), Ccid.to(device), Qwid.to(device), Qcid.to(device)
            p1, p2 = model(Cwid, Ccid, Qwid, Qcid)
            y1, y2 = y1.to(device), y2.to(device)
            loss1 = F.cross_entropy(p1, y1)
            loss2 = F.cross_entropy(p2, y2)
            loss = loss1 + loss2
            losses.append(loss.item())
            y1_ = torch.argmax(p1, 1)
            y2_ = torch.argmax(p2, 1)
            answer_dict_, _ = convert_tokens(eval_file, ids.tolist(), y1_.tolist(), y2_.tolist())
            answer_dict.update(answer_dict_)
    loss = np.mean(losses)
    metrics = evaluate(eval_file, answer_dict)
    f = open("log/ans_{}.json".format(epoch), "w")
    json.dump(answer_dict, f)
    f.close()
    metrics["loss"] = loss
    print("EPOCH {:8d} loss {:8f} F1 {:8f} EM {:8f}\n".format(epoch, loss, metrics["f1"], metrics["exact_match"]))
    return metrics


def train_entry(config):
    from models import QANet

    with open(config.word_emb_file, "r") as fh:
        word_mat = np.array(json.load(fh), dtype=np.float32)
    with open(config.char_emb_file, "r") as fh:
        char_mat = np.array(json.load(fh), dtype=np.float32)
    with open(config.dev_eval_file, "r") as fh:
        dev_eval_file = json.load(fh)

    print("Building model...")

    train_dataset = SQuADDataset(config.train_record_file, config.num_steps, config.batch_size)
    dev_dataset = SQuADDataset(config.dev_record_file, config.val_num_batches, config.batch_size)

    lr = config.learning_rate

    model = QANet(word_mat, char_mat).to(device)
    parameters = filter(lambda param: param.requires_grad, model.parameters())
    optimizer = optim.Adam(betas=(0.8, 0.999), eps=1e-7, weight_decay=3e-7, params=parameters)
    crit = lr / math.log2(1000)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda ee: crit * math.log2(
        ee + 1) if ee + 1 <= 1000 else lr)
    L = config.checkpoint
    N = config.num_steps
    best_f1 = 0
    best_em = 0
    for ep in range(L, N + L, L):
        train(model, scheduler, train_dataset, ep, L)
        metrics = test(model, dev_dataset, dev_eval_file, ep)
        dev_f1 = metrics["f1"]
        dev_em = metrics["exact_match"]
        if dev_f1 < best_f1 and dev_em < best_em:
            patience += 1
            if patience > config.early_stop:
                break
        else:
            patience = 0
            best_f1 = max(best_f1, dev_f1)
            best_em = max(best_em, dev_em)

        fn = os.path.join(config.save_dir, "model_{}.ckpt".format(ep))
        torch.save(model, fn)


def test_entry(config):
    pass


def main(_):
    if config.mode == "train":
        train_entry(config)
    elif config.mode == "data":
        preproc(config)
    elif config.mode == "debug":
        config.num_steps = 4
        config.val_num_batches = 2
        config.checkpoint = 2
        config.period = 1
        train_entry(config)
    elif config.mode == "test":
        test_entry(config)
    else:
        print("Unknown mode")
        exit(0)


if __name__ == '__main__':
    app.run(main)
