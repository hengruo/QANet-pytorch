import torch
import torch.nn as nn
import torch.nn.functional as F
import config

char_limit = config.char_limit
char_dim = config.char_dim
char_hidden_size = config.char_hidden_size
char_num_layers = config.char_num_layers
char_dir = config.char_dir

dropout = config.dropout
batch_size = config.batch_size
hidden_size = config.hidden_size
word_emb_size = config.word_emb_size
char_emb_size = config.char_emb_size
emb_size = config.emb_size

device = config.device

class QANet(nn.Module):
    pass