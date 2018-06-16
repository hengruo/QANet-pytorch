# QANet-pytorch

## NOTICE
This repo is under re-implementation. Due to frequent modification, this code may not run normally.

Any contributions are welcome!

## Introduction

An implementation of [QANet](https://arxiv.org/pdf/1804.09541.pdf) with PyTorch.

Now it can reach EM/F1 = 70.5/77.2 after 20 epoches for about 20 hours on one 1080Ti card.  

## Usage

Python 3.6 & PyTorch 0.4

1. Install pytorch 0.4 for Python 3.6+
2. Run `pip install spacy tqdm ujson requests`
3. Run `download.sh`
4. Run `python main.py --mode data`
5. Run `python main.py --mode train`

## Structure
dataset.py: download dataset and parse.

main.py: program entry.

models.py: QANet structure.

## Differences from the paper

1. The paper doesn't mention which activation function they used. I use relu.
2. I don't set the embedding of `<UNK>` trainable.
3. The connector between embedding layers and embedding encoders may be different from the implementation of Google, since the description in the paper is inconsistent (residual block can't be used because the dimensions of input and output are different) and they don't say how they implement it.
4. Max passage length is 300 instead of 400 since I don't have much GPU memory.

## TODO

- [ ] Reduce memory usage
- [ ] Performance analysis
- [ ] Reach state-of-art scroes of the original paper
- [ ] Ablation analysis

## Contributors
[InitialBug](https://github.com/InitialBug): find a bug.