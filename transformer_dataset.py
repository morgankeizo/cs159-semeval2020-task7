#!/usr/bin/env python

from glob import glob

import torch
from torch.utils.data import Dataset, random_split


def split_dataset(dataset, alpha):
    train_size = int((1 - alpha) * len(dataset))
    test_size = len(dataset) - train_size
    return random_split(dataset, [train_size, test_size])


class TransformerDataset(Dataset):
    def __init__(self, transformer_cache, transform, wordnet_cache=None):
        self.transform, self.uses_wordnet = get_transform(transform)
        self.transformer_cache = transformer_cache
        self.transformer_chunks = list(glob(f"{transformer_cache}/chunk_*.pt"))
        self.transformer_chunks.sort()
        if self.uses_wordnet:
            self.wordnet_cache = wordnet_cache
            self.wordnet_chunks = list(glob(f"{wordnet_cache}/chunk_*.pt"))
            self.wordnet_chunks.sort()

    def __len__(self):
        return len(self.transformer_chunks)

    def __getitem__(self, key):
        _, _, X1, X2, y = torch.load(self.transformer_chunks[key])
        if self.uses_wordnet:
            _, _, similarities = torch.load(self.wordnet_chunks[key])
            X = self.transform(X1, X2, similarities)
        else:
            X = self.transform(X1, X2)
        return X, y


def transform_default(X1, X2):
    return torch.cat([X1, X2], dim=-1)


def transform_duluth(X1, X2):
    return torch.cat([X1, X2, (X1 - X2).abs(), X1 * X2], dim=-1)


def _transform_wordnet_all(similarities):
    s = list(similarities.values())
    s = [col.nan_to_num(0, 0, 0) for col in s]
    return torch.stack(s).transpose(0, 1)


def _transform_wordnet_key(key, similarities):
    col = similarities[key]
    col = col.nan_to_num(0, 0, 0)
    return col.reshape(col.shape[0], 1)


def _get_transform_one(transform_name):
    """Given a single transform name, return transform and uses_wordnet"""
    if transform_name == "default":
        return transform_default, False
    if transform_name == "duluth":
        return transform_duluth, False
    if transform_name == "wordnet_all":
        return _transform_wordnet_all, True
    if "wordnet_" in transform_name:
        def transform(similarities):
            return _transform_wordnet_key(transform_name[8:], similarities)
        return transform, True


def get_transform(transform_name):
    """Given a full transform name, return transform and uses_wordnet"""

    mapped = list(map(_get_transform_one, transform_name.split("+")))
    _, uses_wordnets = zip(*mapped)

    def transform(X1, X2, similarities=None):
        nonlocal mapped
        Xs = [(fn(similarities) if uses_wordnet else fn(X1, X2))
              for fn, uses_wordnet in mapped]
        return torch.cat(Xs, dim=-1)

    return transform, any(uses_wordnets)
