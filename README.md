# QANet-pytorch

# Introduction

An implementation of [QANet](https://arxiv.org/pdf/1804.09541.pdf) with PyTorch. Now it can reach EM/F1 = 70.5/77.2 after 20 epoches.

## Usage

Python 3.6 & PyTorch 0.4

1. Install pytorch 0.4 for Python 3.6+
2. Run `pip install spacy tqdm ujson requests`
3. Run `python -m spacy download en`
4. Run `python main.py`

## Structure
dataset.py: download dataset and parse.

main.py: program entry.

models.py: QANet structure.

## Differences from the paper

1. The paper doesn't mention which activation function they used. I use relu.
2. I don't set the embedding of `<UNK>` trainable.
3. The connector between embedding layers and embedding encoders may be different from the implementation of Google, since the description in the paper is inconsistent (residual block can't be used because the dimensions of input and output are different) and they don't say how they implement it.

## TODO
- [ ] Reach state-of-art scroes of the original paper
- [ ] Ablation analysis
- [ ] Improvement
