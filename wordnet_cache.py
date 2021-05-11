#!/usr/bin/env python

import argparse
import math
import os
import time

import pandas as pd
import torch

from wordnet import load_nltk, get_similarities, get_text


def main(args):
    if not os.path.exists(args.wordnet_cache):
        os.makedirs(args.wordnet_cache)

    load_nltk(args.nltk_cache)
    df = pd.read_csv(args.data)
    pad = math.ceil(math.log10(len(df)))

    skipped = 0

    def count_skipped(*args):
        nonlocal skipped
        skipped += 1

    for offset in range(0, len(df), args.batch_size):
        offset_next = min(offset + args.batch_size, len(df))
        df_chunk = df[offset:offset_next]

        time1 = time.time()
        similarities = get_similarities(*get_text(df_chunk),
                                        error_callback=count_skipped)
        obj = (offset, offset_next, similarities)
        delta = time.time() - time1

        torch.save(obj, f"{args.wordnet_cache}/chunk_{offset:0{pad}d}.pt")
        print((f"rows {offset:{pad}d} to {offset_next:{pad}d} | "
               f"time {delta:.3f}s"))

    print(f"skipped {skipped} rows")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=str)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--nltk_cache", type=str, default="cache/nltk")
    parser.add_argument("--wordnet_cache", type=str, default="cache/wordnet")
    args = parser.parse_args()
    main(args)
