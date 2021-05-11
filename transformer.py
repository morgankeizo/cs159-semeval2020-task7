#!/usr/bin/env python

import re

from allennlp.modules.scalar_mix import ScalarMix
import torch
from transformers import (BertTokenizer, BertModel,
                          RobertaTokenizer, RobertaModel,
                          XLNetTokenizer, XLNetModel)

REGEX_REPLACE = re.compile("<.*/>")

bert_dict = {"bert": (BertTokenizer, BertModel),
             "roberta": (RobertaTokenizer, RobertaModel),
             "xlnet": (XLNetTokenizer, XLNetModel)}


def load_bert(model_name, cache_dir):
    Tokenizer, Model = bert_dict[model_name.split("-")[0]]
    return (
        Tokenizer.from_pretrained(model_name, cache_dir=cache_dir),
        Model.from_pretrained(model_name, cache_dir=cache_dir))


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
        self.scalar_mix = ScalarMix(
            transformer.config.num_hidden_layers + 1,
            do_layer_norm=False)

    def __call__(self, df):
        """Converts a dataframe into a matrix"""

        text_masked, text_edited = self.get_text(df)
        X_masked = self.text_to_vec(text_masked)
        X_edited = self.text_to_vec(text_edited, pool=True)
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
            text_mask = REGEX_REPLACE.sub(self.tokenizer.mask_token, text)
            text_edit = REGEX_REPLACE.sub(edit, text)
            text_masked.append(text_mask)
            text_edited.append(text_edit)
        return text_masked, text_edited

    def text_to_vec(self, text, pool=False):
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
                                  attention_mask=encoding.attention_mask,
                                  output_hidden_states=True)
        mask_finder = encoding.input_ids == self.tokenizer.mask_token_id
        hidden_state = self.scalar_mix(output.hidden_states)

        if pool:
            # Mean pool sequence except mask (edit)
            span_mask = (mask_finder.logical_not() &
                         encoding.attention_mask.bool())
            return mean_pool(hidden_state, span_mask)
        else:
            # Grab the mask (context)
            return hidden_state[mask_finder]
