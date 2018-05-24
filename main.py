from config import config
from preproc import preproc
from absl import app
import math
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
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader


class SQuADDataset(Dataset):
    def __init__(self, npz_file, num_steps, batch_size):
        data = np.load(npz_file)
        self.context_idxs = torch.Tensor(data["context_idxs"]).long()
        self.context_char_idxs = torch.Tensor(data["context_char_idxs"]).long()
        self.ques_idxs = torch.Tensor(data["ques_idxs"]).long()
        self.ques_char_idxs = torch.Tensor(data["ques_char_idxs"]).long()
        self.y1s = torch.Tensor(data["y1s"]).long()
        self.y2s = torch.Tensor(data["y2s"]).long()
        self.ids = torch.Tensor(data["ids"]).long()
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
        idxs = torch.Tensor(self.idx_map[item:item+self.batch_size]).long()
        res = (self.context_idxs[idxs], self.context_char_idxs[idxs], self.ques_idxs[idxs], self.ques_char_idxs[idxs], self.y1s[idxs],
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


def train(config):
    from models import QANet

    with open(config.word_emb_file, "r") as fh:
        word_mat = np.array(json.load(fh), dtype=np.float32)
    with open(config.char_emb_file, "r") as fh:
        char_mat = np.array(json.load(fh), dtype=np.float32)
    with open(config.train_eval_file, "r") as fh:
        train_eval_file = json.load(fh)
    with open(config.dev_eval_file, "r") as fh:
        dev_eval_file = json.load(fh)
    with open(config.dev_meta, "r") as fh:
        meta = json.load(fh)

    dev_total = meta["total"]
    print("Building model...")

    train_dataset = SQuADDataset(config.train_record_file, config.num_steps, config.batch_size)
    dev_dataset = SQuADDataset(config.dev_record_file, config.num_steps, config.batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lr = config.learning_rate

    model = QANet(word_mat, char_mat).to(device)
    parameters = filter(lambda param: param.requires_grad, model.parameters())
    optimizer = optim.Adam(betas=(0.8, 0.999), eps=1e-7, weight_decay=3e-7, params=parameters)
    crit = lr / math.log2(1000)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda ee: crit * math.log2(
        ee + 1) if ee + 1 <= 1000 else lr)

    for ep in tqdm(range(config.num_steps)):
        (Cwid, Ccid, Qwid, Qcid, y1, y2, ids) = train_dataset[ep]
        p1, p2 = model(Cwid.to(device), Ccid.to(device), Qwid.to(device), Qcid.to(device))
        y1, y2 = y1.to(device), y2.to(device)
        loss1 = F.cross_entropy(p1, y1)
        loss2 = F.cross_entropy(p2, y1)
        loss = loss1 + loss2
        loss.backward()
        scheduler.step()

def test(config):
    pass


def main(_):
    if config.mode == "train":
        train(config)
    elif config.mode == "data":
        preproc(config)
    elif config.mode == "debug":
        config.num_steps = 2
        config.val_num_batches = 1
        config.checkpoint = 1
        config.period = 1
        train(config)
    elif config.mode == "test":
        test(config)
    else:
        print("Unknown mode")
        exit(0)


if __name__ == '__main__':
    app.run(main)
