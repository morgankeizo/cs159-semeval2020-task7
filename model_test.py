#!/usr/bin/env python

import argparse

import numpy as np
import torch

from model import MLP
from transformer_dataset import TransformerDataset


def main(args):

    # load model
    transform, state_dict, model_args = torch.load(args.model)
    model = MLP(*model_args)
    model.load_state_dict(state_dict)

    # load dataset
    dataset = TransformerDataset(args.transformer_cache, transform=transform)

    # test model
    criterion = torch.nn.MSELoss()

    model.eval()
    square_error, size = 0, 0
    with torch.no_grad():
        for X_test, y_test in dataset:
            y_pred = model(X_test).squeeze()
            loss = criterion(y_pred, y_test)
            square_error += loss.item() * len(y_test)
            size += len(y_test)

    rmse = np.sqrt(square_error / size)
    print(f"rmse {rmse}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=str)
    parser.add_argument("--transformer_cache", type=str,
                        default="cache/transformer")
    args = parser.parse_args()
    main(args)
