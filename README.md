# cs159-semeval2020-task7

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
