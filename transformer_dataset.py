#!/usr/bin/env python

import glob

import torch
from torch.utils.data import Dataset, random_split


def split_dataset(dataset, alpha):
    train_size = int((1 - alpha) * len(dataset))
    test_size = len(dataset) - train_size
    return random_split(dataset, [train_size, test_size])


class TransformerDataset(Dataset):
    def __init__(self, transformer_cache, transform):
        self.transformer_cache = transformer_cache
        self.transform = transform_dict[transform]
        self.chunks = list(glob.glob(f"{transformer_cache}/chunk_*.pt"))
        self.chunks.sort()

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, key):
        _, _, X1, X2, y = torch.load(self.chunks[key])
        X = self.transform(X1, X2)
        return X, y


def transform_default(X1, X2):
    return torch.cat([X1, X2], dim=-1)


def transform_duluth(X1, X2):
    return torch.cat([X1, X2, (X1 - X2).abs(), X1 * X2], dim=-1)


transform_dict = {"default": transform_default,
                  "duluth": transform_duluth}
transform_values = list(transform_dict.keys())
