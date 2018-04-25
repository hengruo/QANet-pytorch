# ZZXproject

## Usage

Python 3.5/3.6 & PyTorch 0.3.1

1. Install pytorch 0.3.1 for Python 3.5+
2. Run `pip3 install spacy tqdm ujson requests`
3. Run `python3 main.py`

## Structure
dataset.py: download dataset and parse.

main.py: program entry.

models.py: R-net structure.

error_analysis.py: analyze error answers

## Checkpoints
**20180226**

Implementation of R-net in PyTorch without self-attention.

[Natural Language Computing Group, MSRA: R-NET: Machine Reading Comprehension with Self-matching Networks](https://www.microsoft.com/en-us/research/publication/mrc/)


**20180326**

Complete implementation.

**20180401**

TODO: new embedding for numbers.