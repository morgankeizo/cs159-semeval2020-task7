# cs159-semeval2020-task7

## Introduction

This final project for CS159 Natural Language Processing at Harvey Mudd College
is a solution to [SemEval 2020 Task 7](https://arxiv.org/abs/2008.00304), which
rates the funniness (from 0-3) of edits to news headlines, such as:

> Mitch McConnell thinks tax reform will take longer than Trump ~~claimed~~
**haircut**

This implementation is based on the
[Duluth](https://www.aclweb.org/anthology/2020.semeval-1.128/) team's solution,
which uses the RoBERTa embeddings of the original headlines, the edited
headlines,  as well as vector differences to represent the difference between
original and edited headlines. This project expands on the Duluth implementation
by using various word similarity metrics based on the hypothesis that when the
edited word is dissimilar from the original word in the headline, the edit is
funnier.

## Setup

Download the dataset, install Python dependencies, and activate the
`virtualenv` with the following:

```sh
make data deps
source venv/bin/activate
```

## Run

### Transformer caching

Since experiments will share the same embeddings for its derived features, we
save the embeddings of the training data to a "transformer cache" of chunks of
size `batch_size`. The same follows for the testing data.

```sh
python transformer_cache.py data/subtask-1/train.csv --batch_size 100 \
    --transformer_cache cache/transformer-roberta-train --model_name roberta-base
python transformer_cache.py data/subtask-1/test.csv --batch_size 100 \
    --transformer_cache cache/transformer-roberta-test --model_name roberta-base
```

### Training

Train and save a model by a named transformer (such as `duluth`) with:

```sh
python model_train.py --transform duluth --epochs 10 \
    --save roberta_duluth_10.pt --plot roberta_duluth_10.png \
    --transformer_cache cache/transformer-roberta-train
```

### Testing

Test a saved model with:

```sh
python model_test.py roberta_duluth_10.pt \
    --transformer_cache cache/transformer-roberta-test
```

### Knowledge-based similarity metrics

Similar to the word embeddings previously, we also cache the knowledge-based
(WordNet-based) similarity metrics with:

```sh
python wordnet_cache.py data/subtask-1/train.csv --batch_size 100 \
     --wordnet_cache cache/wordnet-train
python wordnet_cache.py data/subtask-1/test.csv --batch_size 100 \
     --wordnet_cache cache/wordnet-test
```

And can include them in concatenated transforms (such as `duluth+wordnet_lch`,
which uses Duluth features and the Leacock-Chodorow similarity metric):

```sh
python model_train.py --transform duluth+wordnet_lch --epochs 10 \
    --save combined.pt --plot combined.png \
    --transformer_cache cache/transformer-roberta-train \
    --wordnet_cache cache/wordnet-train
python model_test.py combined.pt \
    --transformer_cache cache/transformer-roberta-test \
    --wordnet_cache cache/wordnet-test
```
