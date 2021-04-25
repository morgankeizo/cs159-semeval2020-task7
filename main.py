#!/usr/bin/env python

import re

import pandas as pd
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel


def mean_pool(sequence, mask):
    mask = mask.unsqueeze(dim=-1)
    sequence = sequence.masked_fill(mask == 0, 0)
    return sequence.sum(dim=1) / mask.sum(dim=1).float()


def get_mean_grade(train_i):
    return torch.as_tensor(train_i["meanGrade"].values, dtype=torch.float32)


class Transformer():
    def __init__(self, tokenizer, transformer):
        self.tokenizer = tokenizer
        self.transformer = transformer

    def __call__(self, train_i):
        self.preprocess(train_i)
        text1 = train_i["original_mask"].tolist()
        text2 = train_i["original_edit"].tolist()
        X1 = self.text_to_vec(text1)
        X2 = self.text_to_vec(text2)
        X = torch.cat([X1, X2, (X1 - X2).abs(), X1 * X2], dim=-1)
        return X

    def preprocess(self, train):
        """
        Add new columns to train data frame
        - original_mask: <word/> replaced with [MASK]
        - original_edit: <word/> replaced with edit word
        """

        original_mask = []
        original_edit = []
        for text, edit in zip(train["original"], train["edit"]):
            text_mask = re.sub("<.*/>", self.tokenizer.mask_token, text)
            text_edit = re.sub("<.*/>", edit, text)
            original_mask.append(text_mask)
            original_edit.append(text_edit)
        train["original_mask"] = original_mask
        train["original_edit"] = original_edit

    def text_to_vec(self, text):
        """
        Converts a batch of headlines into a feature matrix

        Text is expected to be preprocessed

        Text is a batch of headlines, like
            ['France is ‘ hunting down its citizens who joined [MASK] ’ without
              trial in Iraq',
             'Pentagon claims 2,000 % increase in Russian trolls after [MASK]
              strikes . What does that mean ?']

        Because we use batch with padding, this tokenizes into
            tensor([[  101,  2605,  2003,  1520,  5933,  2091,  2049,  4480,
                      2040,  2587,   103,  1521,  2302,  3979,  1999,  5712,
                       102,     0,     0,     0,     0],
                    [  101, 20864,  4447,  1016,  1010,  2199,  1003,  3623,
                      1999,  2845, 27980,  2044,   103,  9326,  1012,  2054,
                      2515,  2008,  2812,  1029,   102]]

        With the attention mask masking padded elements
        """

        encoding = self.tokenizer(text, return_tensors="pt", padding=True)
        output = self.transformer(encoding.input_ids,
                                  attention_mask=encoding.attention_mask)
        mask_mask = encoding.input_ids != self.tokenizer.mask_token_id
        span_mask = mask_mask & encoding.attention_mask.bool()
        return mean_pool(output.last_hidden_state, span_mask)


class MLP(nn.Module):
    def __init__(self, input_dimensions, hidden_dimensions=256):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dimensions, hidden_dimensions),
            nn.Tanh(),
            nn.LayerNorm(hidden_dimensions),
            nn.Dropout(0.4),
            nn.Linear(hidden_dimensions, 1),
        )

    def forward(self, x):
        return self.classifier(x)


# loading
train = pd.read_csv("data/task-1/train.csv")

# bert
cache_dir = "bert-cache"
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
transformer = BertModel.from_pretrained(model_name, cache_dir=cache_dir)

# transform
headline_transformer = Transformer(tokenizer, transformer)

# todo: transform/train over BucketIterator
df_train = train[:100].copy()
df_test = train[100:150].copy()
X_train, y_train = headline_transformer(df_train), get_mean_grade(df_train)
X_test, y_test = headline_transformer(df_test), get_mean_grade(df_test)

# train classifier
model = MLP(X_train.shape[1])
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, eps=1e-8)

model.train()
for epoch in range(20):
    optimizer.zero_grad()
    y_pred = model(X_train).squeeze()
    loss = torch.sqrt(criterion(y_pred, y_train))
    print(f"Epoch {epoch} | Loss {loss.item()}")
    loss.backward(retain_graph=True)
    optimizer.step()

# test classifier
y_pred = model(X_test).squeeze()
rmse = torch.sqrt(criterion(y_pred, y_test))
print(rmse.item())
