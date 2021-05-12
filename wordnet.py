#!/usr/bin/env python

import re

import nltk
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from nltk.wsd import lesk
import torch

REGEX_SEARCH = re.compile("<(.*)/>")
REGEX_REPLACE = re.compile("<.*/>")

similarity_names = ["lch", "wup", "res_brown", "lin_brown", "jcn_brown",
                    "pmi_brown"]

wn, ics, finders = None, None, None
measures = BigramAssocMeasures()


def load_nltk(nltk_cache):
    global wn, ics, finders
    nltk.download("wordnet", download_dir=nltk_cache)
    nltk.download("wordnet_ic", download_dir=nltk_cache)
    nltk.download("brown", download_dir=nltk_cache)
    nltk.data.path.append(nltk_cache)
    from nltk.corpus import (wordnet as wn,
                             wordnet_ic as wn_ic,
                             brown)
    ics = {"brown": wn_ic.ic("ic-brown.dat")}
    finders = {"brown": BigramCollocationFinder.from_words(brown.words(), 10)}


def lesk_match_ok(os, es):
    return os and es and (os.pos() == es.pos())


def lesk_match(original_word, original_sentence,
               edit_word, edit_sentence):
    """
    Use Lesk to get original and edit synsets

    Try until both exist with the same POS
    """

    os = lesk(original_sentence, original_word)
    es = lesk(edit_sentence, edit_word)
    if lesk_match_ok(os, es):
        return os, es
    if os:
        es2 = lesk(edit_sentence, edit_word, pos=os.pos())
        if lesk_match_ok(os, es2):
            return os, es2
    if es:
        os2 = lesk(original_sentence, original_word, pos=es.pos())
        if lesk_match_ok(os2, es):
            return os2, es
    return None, None


def get_similarities(original_words, original_sentences,
                     edit_words, edit_sentences, error_callback=None):

    s = {k: [] for k in similarity_names}

    for (original_word, original_sentence,
         edit_word, edit_sentence) in zip(original_words, original_sentences,
                                          edit_words, edit_sentences):
        os, es = lesk_match(original_word, original_sentence,
                            edit_word, edit_sentence)
        try:
            similarities = [
                os.lch_similarity(es),
                os.wup_similarity(es),
                os.res_similarity(es, ics["brown"]),
                os.lin_similarity(es, ics["brown"]),
                os.jcn_similarity(es, ics["brown"]),
                finders["brown"].score_ngram(measures.pmi,
                                             original_word, edit_word) or 0
            ]
        except:
            similarities = [0] * len(similarity_names)
            if error_callback:
                error_callback(original_word, original_sentence,
                               edit_word, edit_sentence)

        for sk, sv in zip(similarity_names, similarities):
            s[sk].append(sv)

    return {k: torch.tensor(v, dtype=torch.float32) for k, v in s.items()}


def get_text(df):

    original_words, original_sentences = [], []
    edit_words, edit_sentences = [], []

    for text, edit in zip(df["original"], df["edit"]):
        original_word = REGEX_SEARCH.search(text).group(1)
        original_sentence = REGEX_REPLACE.sub(original_word, text)
        edit_sentence = REGEX_REPLACE.sub(edit, text)

        original_words.append(original_word)
        original_sentences.append(original_sentence)
        edit_words.append(edit)
        edit_sentences.append(edit_sentence)

    return (original_words, original_sentences,
            edit_words, edit_sentences)
