#!/usr/bin/env python

import argparse
import math
import os
import psutil
import time

import pandas as pd
import torch

from transformer import load_bert, get_mean_grade, Transformer


def get_memory_mb():
    return psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2


def main(args):
    if not os.path.exists(args.transformer_cache):
        os.makedirs(args.transformer_cache)

    bert = load_bert(args.model_name, args.bert_cache)
    transformer = Transformer(*bert)
    df = pd.read_csv(args.data)
    pad = math.ceil(math.log10(len(df)))

    for offset in range(0, len(df), args.batch_size):
        offset_next = min(offset + args.batch_size, len(df))
        df_chunk = df[offset:offset_next]

        time1 = time.time()
        X1, X2 = transformer(df_chunk)
        y = get_mean_grade(df_chunk)
        obj = (offset, offset_next, X1, X2, y)
        delta = time.time() - time1

        torch.save(obj, f"{args.transformer_cache}/chunk_{offset:0{pad}d}.pt")
        print((f"rows {offset:{pad}d} to {offset_next:{pad}d} | "
               f"time {delta:.3f}s mem {get_memory_mb():.3f} MB"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=str)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--transformer_cache", type=str,
                        default="transformer-cache")
    parser.add_argument("--model_name", type=str, default="bert-base-uncased")
    parser.add_argument("--bert_cache", type=str, default="bert-cache")
    args = parser.parse_args()
    main(args)
