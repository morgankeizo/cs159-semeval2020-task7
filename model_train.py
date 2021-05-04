#!/usr/bin/env python

import argparse
import time

import numpy as np
import matplotlib.pyplot as plt
import torch

from model import MLP
from transformer_dataset import TransformerDataset, split_dataset


def make_plot(data, save):
    plt.xlabel("epoch")
    plt.ylabel("rmse")
    plt.plot(data["epoch"], data["loss_train"], label="train")
    if data["loss_test"]:
        plt.plot(data["epoch"], data["loss_test"], label="test")
    plt.legend()
    plt.savefig(save)


def main(args):

    # load dataset
    dataset = TransformerDataset(args.transformer_cache,
                                 transform=args.transform)
    width = dataset[0][0].shape[1]
    dataset_train, dataset_test = (split_dataset(dataset, args.alpha)
                                   if args.alpha else (dataset, []))

    # train classifier
    model_args = (width, args.hidden_dimensions, args.dropout)
    model = MLP(*model_args)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, eps=args.eps)

    epoch_rmse_train = []
    epoch_rmse_test = []
    for epoch in range(args.epochs):
        time1 = time.time()

        # train
        model.train()
        square_error_train, size_train = 0, 0
        for X_train, y_train in dataset_train:
            optimizer.zero_grad()
            y_pred = model(X_train).squeeze()
            loss = criterion(y_pred, y_train)
            loss.backward()
            optimizer.step()
            square_error_train += loss.item() * len(y_train)
            size_train += len(y_train)

        delta = time.time() - time1
        rmse_train = np.sqrt(square_error_train / size_train)
        epoch_rmse_train.append(rmse_train)

        # test
        if dataset_test:
            model.eval()
            square_error_test, size_test = 0, 0
            with torch.no_grad():
                for X_test, y_test in dataset_test:
                    y_pred = model(X_test).squeeze()
                    loss = criterion(y_pred, y_test)
                    square_error_test += loss.item() * len(y_test)
                    size_test += len(y_test)

            rmse_test = np.sqrt(square_error_test / size_test)
            epoch_rmse_test.append(rmse_test)

        rmse_test_str = f"{rmse_test:.6f}" if dataset_test else "--"
        print((f"epoch {epoch} "
               f"loss train {rmse_train:.6f} "
               f"test {rmse_test_str} "
               f"time {delta:.3f}"))

    if args.plot:
        data = {"epoch": list(range(args.epochs)),
                "loss_train": epoch_rmse_train,
                "loss_test": epoch_rmse_test}
        make_plot(data, args.plot)

    if args.save:
        obj = (args.transform, model.state_dict(), model_args)
        torch.save(obj, args.save)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--transformer_cache", type=str,
                        default="cache/transformer")
    parser.add_argument("--transform", type=str, default="default")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--hidden_dimensions", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--plot", type=str)
    parser.add_argument("--save", type=str)
    args = parser.parse_args()
    main(args)
