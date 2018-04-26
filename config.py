import torch

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