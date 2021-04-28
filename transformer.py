#!/usr/bin/env python

import re

import torch
from transformers import BertTokenizer, BertModel


def load_bert(model_name="bert-base-uncased", cache_dir="bert-cache"):
    return (
        BertTokenizer.from_pretrained(model_name, cache_dir=cache_dir),
        BertModel.from_pretrained(model_name, cache_dir=cache_dir))


def get_mean_grade(df):
    return torch.as_tensor(df["meanGrade"].values, dtype=torch.float32)


def mean_pool(sequence, mask):
    mask = mask.unsqueeze(dim=-1)
    sequence = sequence.masked_fill(mask == 0, 0)
    return sequence.sum(dim=1) / mask.sum(dim=1).float()


class Transformer():
    def __init__(self, tokenizer, transformer):
        self.tokenizer = tokenizer
        self.transformer = transformer

    def __call__(self, df):
        """Converts a dataframe into a matrix"""

        text_masked, text_edited = self.get_text(df)
        X_masked = self.text_to_vec(text_masked)
        X_edited = self.text_to_vec(text_edited)
        return X_masked, X_edited

    def get_text(self, df):
        """
        Get two mappings of text from csv
        - text_masked: <word/> replaced with [MASK]
        - text_edited: <word/> replaced with edit word
        """

        text_masked = []
        text_edited = []
        for text, edit in zip(df["original"], df["edit"]):
            text_mask = re.sub("<.*/>", self.tokenizer.mask_token, text)
            text_edit = re.sub("<.*/>", edit, text)
            text_masked.append(text_mask)
            text_edited.append(text_edit)
        return text_masked, text_edited

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