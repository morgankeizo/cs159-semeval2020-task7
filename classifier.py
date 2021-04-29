#!/usr/bin/env python

import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_dimensions, hidden_dimensions, dropout):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dimensions, hidden_dimensions),
            nn.Tanh(),
            nn.LayerNorm(hidden_dimensions),
            nn.Dropout(dropout),
            nn.Linear(hidden_dimensions, 1),
        )

    def forward(self, x):
        return self.classifier(x)
