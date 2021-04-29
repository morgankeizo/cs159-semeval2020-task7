# cs159-semeval2020-task7

## Setup

Download the dataset, install Python dependencies, and activate the
`virtualenv` with the following:

```sh
make data deps
source venv/bin/activate
```

## Run

Since experiments will share the same BERT embeddings for its derived features,
we save the embeddings of the training data to a "transformer cache" of chunks
of size `batch_size`.

```sh
python transformer_cache.py data/subtask-1/train.csv --batch_size 100 \
    --transformer_cache transformer-cache/roberta --model_name roberta-base
```

Run a training experiment by a named transformer (such as `duluth`) with:

```sh
python main.py --transform duluth --epochs 100 --plot duluth_100.png \
    --transformer_cache transformer-cache/roberta
```
