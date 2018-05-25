import os
import requests
import zipfile
import numpy as np
import ujson as uj
import spacy
from collections import defaultdict
import copy
from models import CharEmbedding, Dword, Dchar
import torch
from tqdm import *

'''
Prepare data
'''

# configurations of data
train_filename = "train-v1.1.json"
dev_filename = "dev-v1.1.json"
char_emb_filename = "glove.840B.300d-char.txt"
word_emb_zip = "glove.840B.300d.zip"
word_emb_filename = "glove.840B.300d.txt"

data_dir = "tmp/squad"
emb_dir = "tmp/embedding"

word_emb_url_base = "http://nlp.stanford.edu/data/"
char_emb_url_base = "https://raw.githubusercontent.com/minimaxir/char-embeddings/master/"
train_url_base = "https://rajpurkar.github.io/SQuAD-explorer/dataset/"
dev_url_base = "https://rajpurkar.github.io/SQuAD-explorer/dataset/"

context_max_L = 300


# GloVe embedding

# SQuAD dataset

class SQuAD:
    class Dataset:
        def __init__(self):
            '''
            length: the length of dataset
            passages: indexes of word embeddings for passages
            questions: indexes of word embeddings for questions
            answers: the positions in `wpassages` of the first and last words in an answer
            pack: [(idx of wpassages, idx of wquestions, idx of answers)]
            '''
            self.length = 0
            self.contexts = []
            self.questions = []
            self.question_ids = []
            self.answers = []
            self.packs = []

    def __init__(self):
        self.word_embedding = np.zeros((1, 1))
        self.char_embedding = np.zeros((1, 1))
        self.train = SQuAD.Dataset()
        self.dev = SQuAD.Dataset()
        self.itow = {}
        self.wtoi = {}
        self.itoc = {}
        self.ctoi = {}

    @classmethod
    def load(cls, dirname: str):
        squad = SQuAD()
        f = open(os.path.join(dirname, "itow.json"), "r")
        itow = uj.load(f)
        for k in itow:
            squad.itow[int(k)] = itow[k]
        f.close()
        f = open(os.path.join(dirname, "wtoi.json"), "r")
        squad.wtoi = uj.load(f)
        f.close()
        f = open(os.path.join(dirname, "itoc.json"), "r")
        itoc = uj.load(f)
        for k in itoc:
            squad.itoc[int(k)] = itoc[k]
        f.close()
        f = open(os.path.join(dirname, "ctoi.json"), "r")
        squad.ctoi = uj.load(f)
        f.close()
        f = open(os.path.join(dirname, "char_embedding.npy"), "rb")
        squad.char_embedding = np.load(f)
        f.close()
        f = open(os.path.join(dirname, "word_embedding.npy"), "rb")
        squad.word_embedding = np.load(f)
        f.close()
        f = open(os.path.join(dirname, "train.json"), "r")
        train = uj.load(f)
        for key in train:
            squad.train.__setattr__(key, train[key])
        f.close()
        f = open(os.path.join(dirname, "dev.json"), "r")
        dev = uj.load(f)
        for key in dev:
            squad.dev.__setattr__(key, dev[key])
        f.close()
        return squad

    def dump(self, dirname: str):
        if not os.path.exists(dirname):
            os.mkdir(dirname)
        f = open(os.path.join(dirname, "itow.json"), "w")
        uj.dump(self.itow, f)
        f.close()
        f = open(os.path.join(dirname, "wtoi.json"), "w")
        uj.dump(self.wtoi, f)
        f.close()
        f = open(os.path.join(dirname, "itoc.json"), "w")
        uj.dump(self.itoc, f)
        f.close()
        f = open(os.path.join(dirname, "ctoi.json"), "w")
        uj.dump(self.ctoi, f)
        f.close()
        f = open(os.path.join(dirname, "char_embedding.npy"), "wb")
        np.save(f, self.char_embedding, allow_pickle=False)
        f.close()
        f = open(os.path.join(dirname, "word_embedding.npy"), "wb")
        np.save(f, self.word_embedding, allow_pickle=False)
        f.close()
        f = open(os.path.join(dirname, "train.json"), "w")
        uj.dump(self.train.__dict__, f)
        f.close()
        f = open(os.path.join(dirname, "dev.json"), "w")
        uj.dump(self.dev.__dict__, f)
        f.close()


def download(urlbase, filename, path):
    url = os.path.join(urlbase, filename)
    if not os.path.exists(os.path.join(path, filename)):
        try:
            print("Downloading file {}...".format(filename))
            r = requests.get(url, stream=True)
            fullname = os.path.join(path, filename)
            with open(fullname, 'wb') as f:
                for chunk in r.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)
        except AttributeError as e:
            print("Download error!")
            raise e


def prepare_data():
    '''
    download squad dataset into `data/squad` and embedding into `data/embedding`
    '''
    dirs = [data_dir, emb_dir]
    for d in dirs:
        if not os.path.exists(d):
            os.makedirs(d)
    download(train_url_base, train_filename, data_dir)
    download(dev_url_base, dev_filename, data_dir)
    download(char_emb_url_base, char_emb_filename, emb_dir)
    if not os.path.exists(os.path.join(emb_dir, word_emb_filename)):
        download(word_emb_url_base, word_emb_zip, emb_dir)
        zip_ref = zipfile.ZipFile(os.path.join(
            emb_dir, word_emb_zip), 'r')
        zip_ref.extractall(emb_dir)
        zip_ref.close()
        os.remove(os.path.join(emb_dir, word_emb_zip))


def parse_data_I(squad: SQuAD):
    cembf = open(os.path.join(emb_dir, char_emb_filename), 'r')
    wembf = open(os.path.join(emb_dir, word_emb_filename), 'r')
    ctoi = defaultdict(lambda: len(ctoi))
    wtoi = defaultdict(lambda: len(wtoi))
    specials = ['<UNK>', '<PAD>', '<SOS>', '<EOS>']
    [ctoi[x] for x in specials]
    [wtoi[x] for x in specials]
    cemb = [np.zeros(Dword) for _ in specials]
    wemb = copy.deepcopy(cemb)
    print('Reading char embeddings')
    for line in cembf.readlines():
        tmp = line.split(' ')
        ctoi[tmp[0]]
        cemb.append(np.array([float(x) for x in tmp[1:]]))
    print('Reading word embeddings')
    for line in wembf.readlines():
        tmp = line.split(' ')
        wtoi[tmp[0]]
        wemb.append(np.array([float(x) for x in tmp[1:]]))
        if len(wtoi) != len(wemb):
            _ = wemb.pop()

    for char in ctoi:
        squad.itoc[ctoi[char]] = char
    for word in wtoi:
        squad.itow[wtoi[word]] = word
    squad.wtoi = dict(wtoi)
    squad.ctoi = dict(ctoi)
    print("Dictionary complete.")
    squad.char_embedding = np.array(cemb)
    squad.word_embedding = np.array(wemb)


def convert_idx(text, tokens):
    current = 0
    spans = []
    for token in tokens:
        current = text.find(token, current)
        if current < 0:
            print("Token {} cannot be found".format(token))
            raise Exception()
        spans.append((current, current + len(token)))
        current += len(token)
    return spans


def parse_train(jsondata, squad: SQuAD, dataset: SQuAD.Dataset):
    tokenizer = spacy.blank('en')
    cid, qid, aid = 0, 0, 0
    for item in jsondata:
        for para in item['paragraphs']:
            context = para['context'].replace("''", '" ').replace("``", '" ')
            context_tokens = [token.text for token in tokenizer(context)]
            if len(context_tokens) > context_max_L: continue
            context_token_wids = [squad.wtoi[tk] if tk in squad.wtoi else 0 for tk in context_tokens]
            spans = convert_idx(context, context_tokens)
            for qa in para['qas']:
                ques = qa['question'].replace("''", '" ').replace("``", '" ')
                ques_tokens = [token.text for token in tokenizer(ques)]
                ques_token_wids = [squad.wtoi[tk] if tk in squad.wtoi else 0 for tk in ques_tokens]
                ques_id = qa['id']
                for ans in qa['answers']:
                    ans_text = ans['text']
                    ans_start = ans['answer_start']
                    ans_end = ans_start + len(ans_text)
                    ans_span = []
                    for idx, span in enumerate(spans):
                        if not (ans_end <= span[0] or ans_start >= span[1]):
                            ans_span.append(idx)
                    ans_pair = (ans_span[0], ans_span[-1])
                    dataset.answers.append(ans_pair)
                    dataset.packs.append((cid, qid, aid))
                    aid += 1
                dataset.questions.append(ques_token_wids)
                dataset.question_ids.append(ques_id)
                qid += 1
            dataset.contexts.append(context_token_wids)
            cid += 1
    dataset.length = len(dataset.packs)

def parse_dev(jsondata, squad: SQuAD, dataset: SQuAD.Dataset):
    tokenizer = spacy.blank('en')
    cid, qid, aid = 0, 0, 0
    for item in jsondata:
        for para in item['paragraphs']:
            context = para['context'].replace("''", '" ').replace("``", '" ')
            context_tokens = [token.text for token in tokenizer(context)]
            if len(context_tokens) > context_max_L: continue
            context_token_wids = [squad.wtoi[tk] if tk in squad.wtoi else 0 for tk in context_tokens]
            spans = convert_idx(context, context_tokens)
            for qa in para['qas']:
                ques = qa['question'].replace("''", '" ').replace("``", '" ')
                ques_tokens = [token.text for token in tokenizer(ques)]
                ques_token_wids = [squad.wtoi[tk] if tk in squad.wtoi else 0 for tk in ques_tokens]
                ques_id = qa['id']
                dataset.answers.append((0,0))
                dataset.packs.append((cid, qid, aid))
                aid += 1
                dataset.questions.append(ques_token_wids)
                dataset.question_ids.append(ques_id)
                qid += 1
            dataset.contexts.append(context_token_wids)
            cid += 1
    dataset.length = len(dataset.packs)


def parse_data_II(squad):
    f = open(os.path.join(data_dir, train_filename), 'r')
    train = uj.load(f)['data']
    f.close()
    f = open(os.path.join(data_dir, dev_filename), 'r')
    dev = uj.load(f)['data']
    print("Parsing train dataset.")
    parse_train(train, squad, squad.train)
    print("Parsing dev dataset.")
    parse_dev(dev, squad, squad.dev)
    print("Parsing done.")


def get_embeddings(inputs, squad, wemb, wtoi):
    ss = []
    for p in inputs:
        ws = []
        for i, n in enumerate(p):
            c = squad.itow[n]
            if c not in wtoi:
                nn = wtoi[c]
                wemb.append(squad.word_embedding[n])
                p[i] = nn
            ws.append(wtoi[c])
        ss.append(ws)
    return ss


def parse_data_III(squad):
    print("Reconstruct embeddings.")
    wtoi = defaultdict(lambda: len(wtoi))
    specials = ['<UNK>', '<PAD>', '<SOS>', '<EOS>']
    [wtoi[x] for x in specials]
    wemb = [np.zeros(Dword) for _ in specials]
    squad.train.contexts = get_embeddings(squad.train.contexts, squad, wemb, wtoi)
    squad.dev.contexts = get_embeddings(squad.dev.contexts, squad, wemb, wtoi)
    squad.train.questions = get_embeddings(squad.train.questions, squad, wemb, wtoi)
    squad.dev.questions = get_embeddings(squad.dev.questions, squad, wemb, wtoi)
    itow = {}
    for word in wtoi:
        itow[wtoi[word]] = word
    charemb = CharEmbedding()
    print('Generating word\'s char-level embeddings')
    wcemb = [np.zeros(Dchar) for _ in specials]
    for i in tqdm(range(4, len(wemb))):
        chars = torch.FloatTensor(
            [squad.char_embedding[squad.ctoi[c]] if c in squad.ctoi else squad.char_embedding[0] for c in
             list(squad.itow[i])])
        chars = torch.unsqueeze(chars, 1)
        wcemb.append(charemb(chars).data.numpy())
    squad.word_embedding = np.array(wemb)
    squad.char_embedding = np.array(wcemb)
    squad.itow = dict(itow)
    squad.wtoi = dict(wtoi)


def get_dataset(shrink):
    if os.path.exists('data/') and len(os.listdir('data/')) > 1:
        print("Found existing data.")
        squad = SQuAD.load('data/')
    else:
        prepare_data()
        squad = SQuAD()
        parse_data_I(squad)
        parse_data_II(squad)
        parse_data_III(squad)
        squad.dump("data/")
    if shrink < context_max_L:
        packs = squad.train.packs
        new_packs = []
        for pack in packs:
            if len(squad.train.contexts[pack[0]]) <= shrink:
                new_packs.append(pack)
        squad.train.packs = new_packs
        packs = squad.dev.packs
        new_packs = []
        for pack in packs:
            if len(squad.dev.contexts[pack[0]]) <= shrink:
                new_packs.append(pack)
        squad.dev.packs = new_packs
    return squad

def test():
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
    squad = SQuAD.load("data/")
    sample = squad.train.packs[:100]
    anss = {}
    for p in sample:
        qid = p[1]
        aid = p[2]
        span = squad.train.answers[aid]
        anss[squad.train.question_ids[qid]] = get_anwser(span[0], span[1], p[0], squad.itow, squad.train)
    uj.dump(anss, open("tmp/squad/sample.json", "w"))

if __name__ == "__main__":
    # squad = SQuAD()
    # parse_data_I(squad)
    # parse_data_II(squad)
    # squad.dump("data/")
    # parse_data_III(squad)
    # squad.dump("data/")
    test()
