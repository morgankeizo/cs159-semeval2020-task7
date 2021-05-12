#!/usr/bin/env python

import argparse

import numpy as np
import pandas as pd


def main(args):
    train_mean_grade = pd.read_csv(args.train)["meanGrade"].mean()
    test_mean_grades = pd.read_csv(args.test)["meanGrade"]
    rmse = np.sqrt(((test_mean_grades - train_mean_grade) ** 2).mean())
    print(f"rmse {rmse}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("train", type=str)
    parser.add_argument("test", type=str)
    args = parser.parse_args()
    main(args)
