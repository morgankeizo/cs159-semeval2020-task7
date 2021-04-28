#!/usr/bin/env python

import argparse
import time

import numpy as np
import torch

from classifier import MLP
from transformer_dataset import (TransformerDataset,
                                 split_dataset, transform_values)


def main(args):

    # load dataset
    dataset = TransformerDataset(args.transformer_cache,
                                 transform=args.transform)
    width = dataset[0][0].shape[1]
    dataset_train, dataset_test = split_dataset(dataset, args.alpha)

    # train classifier
    model = MLP(width)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, eps=args.eps)

    for epoch in range(args.epochs):
        time1 = time.time()

        # train
        model.train()
        optimizer.zero_grad()
        square_error_train, size_train = 0, 0
        for X_train, y_train in dataset_train:
            y_pred = model(X_train).squeeze()
            loss = criterion(y_pred, y_train)
            loss.backward()
            optimizer.step()
            square_error_train += loss.item() * len(y_train)
            size_train += len(y_train)

        delta = time.time() - time1

        # test
        model.eval()
        square_error_test, size_test = 0, 0
        with torch.no_grad():
            for X_test, y_test in dataset_test:
                y_pred = model(X_test).squeeze()
                loss = criterion(y_pred, y_test)
                square_error_test += loss.item() * len(y_test)
                size_test += len(y_test)

        rmse_train = np.sqrt(square_error_train / size_train)
        rmse_test = np.sqrt(square_error_test / size_test)
        print((f"epoch {epoch} "
               f"loss train {rmse_train:.6f} "
               f"test {rmse_test:.6f} "
               f"time {delta:.3f}"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--transformer_cache", type=str,
                        default="transformer-cache")
    parser.add_argument("--transform", type=str, default="default")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--eps", type=float, default=1e-8)
    args = parser.parse_args()
    main(args)
