import os
import absl.flags as flags
import torch
import torch.backends.cudnn as cudnn

'''
The content of this file is mostly copied from https://github.com/HKUST-KnowComp/R-Net/blob/master/config.py
'''

home = os.path.expanduser(".")
train_file = os.path.join(home, "data", "squad", "train-v1.1.json")
dev_file = os.path.join(home, "data", "squad", "dev-v1.1.json")
test_file = os.path.join(home, "data", "squad", "dev-v1.1.json")
glove_word_file = os.path.join(home, "data", "glove", "glove.840B.300d.txt")

target_dir = "data"
event_dir = "log"
save_dir = "model"
answer_dir = "log"
train_record_file = os.path.join(target_dir, "train.npz")
dev_record_file = os.path.join(target_dir, "dev.npz")
test_record_file = os.path.join(target_dir, "test.npz")
word_emb_file = os.path.join(target_dir, "word_emb.json")
char_emb_file = os.path.join(target_dir, "char_emb.json")
train_eval = os.path.join(target_dir, "train_eval.json")
dev_eval = os.path.join(target_dir, "dev_eval.json")
test_eval = os.path.join(target_dir, "test_eval.json")
dev_meta = os.path.join(target_dir, "dev_meta.json")
test_meta = os.path.join(target_dir, "test_meta.json")
word2idx_file = os.path.join(target_dir, "word2idx.json")
char2idx_file = os.path.join(target_dir, "char2idx.json")
answer_file = os.path.join(answer_dir, "answer.json")

if not os.path.exists(target_dir):
    os.makedirs(target_dir)
if not os.path.exists(event_dir):
    os.makedirs(event_dir)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
if not os.path.exists(answer_dir):
    os.makedirs(answer_dir)

flags.DEFINE_string("mode", "train", "train/debug/test")

flags.DEFINE_string("target_dir", target_dir, "")
flags.DEFINE_string("event_dir", event_dir, "")
flags.DEFINE_string("save_dir", save_dir, "")
flags.DEFINE_string("train_file", train_file, "")
flags.DEFINE_string("dev_file", dev_file, "")
flags.DEFINE_string("test_file", test_file, "")
flags.DEFINE_string("glove_word_file", glove_word_file, "")

flags.DEFINE_string("train_record_file", train_record_file, "")
flags.DEFINE_string("dev_record_file", dev_record_file, "")
flags.DEFINE_string("test_record_file", test_record_file, "")
flags.DEFINE_string("word_emb_file", word_emb_file, "")
flags.DEFINE_string("char_emb_file", char_emb_file, "")
flags.DEFINE_string("train_eval_file", train_eval, "")
flags.DEFINE_string("dev_eval_file", dev_eval, "")
flags.DEFINE_string("test_eval_file", test_eval, "")
flags.DEFINE_string("dev_meta", dev_meta, "")
flags.DEFINE_string("test_meta", test_meta, "")
flags.DEFINE_string("word2idx_file", word2idx_file, "")
flags.DEFINE_string("char2idx_file", char2idx_file, "")
flags.DEFINE_string("answer_file", answer_file, "")


flags.DEFINE_integer("glove_char_size", 94, "Corpus size for Glove")
flags.DEFINE_integer("glove_word_size", int(2.2e6), "Corpus size for Glove")
flags.DEFINE_integer("glove_dim", 300, "Embedding dimension for Glove")
flags.DEFINE_integer("char_dim", 64, "Embedding dimension for char")

flags.DEFINE_integer("para_limit", 400, "Limit length for paragraph")
flags.DEFINE_integer("ques_limit", 50, "Limit length for question")
flags.DEFINE_integer("ans_limit", 30, "Limit length for answers")
# flags.DEFINE_integer("test_para_limit", 400, "Limit length for paragraph in test file")
# flags.DEFINE_integer("test_ques_limit", 100, "Limit length for question in test file")
flags.DEFINE_integer("char_limit", 16, "Limit length for character")
flags.DEFINE_integer("word_count_limit", -1, "Min count for word")
flags.DEFINE_integer("char_count_limit", -1, "Min count for char")

flags.DEFINE_integer("capacity", 15000, "Batch size of dataset shuffle")
flags.DEFINE_integer("num_threads", 4, "Number of threads in input pipeline")
flags.DEFINE_boolean("is_bucket", False, "build bucket batch iterator or not")
flags.DEFINE_list("bucket_range", [40, 401, 40], "the range of bucket")

flags.DEFINE_integer("batch_size", 16, "Batch size")
flags.DEFINE_integer("num_steps", 60000, "Number of steps")
flags.DEFINE_integer("checkpoint", 1000, "checkpoint to save and evaluate the model")
flags.DEFINE_integer("period", 100, "period to save batch loss")
flags.DEFINE_integer("val_num_batches", 150, "Number of batches to evaluate the model")
flags.DEFINE_integer("test_num_batches", 150, "Number of batches to evaluate the model")
flags.DEFINE_float("dropout", 0.1, "Dropout prob across the layers")
flags.DEFINE_float("dropout_char", 0.05, "Dropout prob across the layers")
flags.DEFINE_float("grad_clip", 5.0, "Global Norm gradient clipping rate")
flags.DEFINE_float("learning_rate", 0.001, "Learning rate")
flags.DEFINE_integer("lr_warm_up_num", 1000, "Number of warm-up steps of learning rate")
flags.DEFINE_float("ema_decay", 0.9999, "Exponential moving average decay")
flags.DEFINE_float("beta1", 0.8, "Beta 1")
flags.DEFINE_float("beta2", 0.999, "Beta 2")
# flags.DEFINE_float("l2_norm", 3e-7, "L2 norm scale")
flags.DEFINE_integer("early_stop", 10, "Checkpoints for early stop")
flags.DEFINE_integer("connector_dim", 96, "Dimension of connectors of each layer")
flags.DEFINE_integer("num_heads", 2, "Number of heads in multi-head attention")

# Extensions (Uncomment corresponding line in download.sh to download the required data)
glove_char_file = os.path.join(home, "data", "glove", "glove.840B.300d-char.txt")
flags.DEFINE_string("glove_char_file", glove_char_file, "Glove character embedding")
flags.DEFINE_boolean("pretrained_char", False, "Whether to use pretrained char embedding")

fasttext_file = os.path.join(home, "data", "fasttext", "wiki-news-300d-1M.vec")
flags.DEFINE_string("fasttext_file", fasttext_file, "Fasttext word embedding")
flags.DEFINE_boolean("fasttext", False, "Whether to use fasttext")

config = flags.FLAGS

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cudnn.enabled = False