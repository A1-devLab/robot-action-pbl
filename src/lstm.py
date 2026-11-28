"""Small LSTM sequence classifier (PyTorch nn.LSTM) with an sklearn-like API.

LSTM unit (Alpaydin 12.7.4): input / forget / output gates decide what the cell state c_t
keeps, adds and outputs; h_t = o * tanh(c_t). The hidden states are pooled over time
('mean') or the last one is used ('last'), then dropout and a logistic output.
Trained full-batch with Adam on binary cross-entropy + L2 on the weights.

Input X: (n_samples, n_steps, n_channels). Channels are standardized with the training data
inside fit(), and augmentation (augment_sequences) is applied to the training data only.
"""

from __future__ import annotations

import copy

import numpy as np
import torch
from torch import nn


def augment_sequences(X, y, n_copies=4, warp=0.10, scale=0.03, noise=0.02, seed=0):
    """Return X, y plus n_copies augmented copies of every sequence.

    warp:  random monotone time warp, time grid moved by up to +-warp of the length
    scale: per-channel amplitude factor 1 +- scale
    noise: Gaussian noise, std = noise * channel std
    """
    if n_copies <= 0:
        return X, y
    rng = np.random.default_rng(seed)
    n, steps, channels = X.shape
    grid = np.linspace(0, 1, steps)
    channel_std = X.reshape(-1, channels).std(axis=0) + 1e-12
    copies = [X]
    for _ in range(n_copies):
        out = np.empty_like(X)
        for k in range(n):
            # smooth monotone warp: a sine bump with random amplitude, end points fixed
            warped = grid + rng.uniform(-warp, warp) * np.sin(np.pi * grid) / np.pi
            warped = np.clip(np.maximum.accumulate(warped), 0, 1)
            for c in range(channels):
                out[k, :, c] = np.interp(warped, grid, X[k, :, c])
            out[k] *= 1 + rng.uniform(-scale, scale, size=channels)
            out[k] += rng.normal(0, noise, size=(steps, channels)) * channel_std
        copies.append(out)
    return np.concatenate(copies), np.tile(y, n_copies + 1)


class LSTMNet(nn.Module):
    def __init__(self, n_channels, hidden, dropout, pooling):
        super().__init__()
        self.lstm = nn.LSTM(n_channels, hidden, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(hidden, 1)
        self.pooling = pooling
        with torch.no_grad():                    # gate order i, f, g, o: forget gate starts open
            self.lstm.bias_ih_l0.zero_()
            self.lstm.bias_hh_l0.zero_()
            self.lstm.bias_ih_l0[hidden:2 * hidden] = 1.0

    def forward(self, x):
        h, _ = self.lstm(x)                      # (batch, steps, hidden)
        pooled = h.mean(dim=1) if self.pooling == "mean" else h[:, -1]
        return self.out(self.drop(pooled)).squeeze(1)


class LSTMClassifier:
    """pooling: 'last' uses h_T, 'mean' averages h_t over all steps.
    val_fraction > 0: early stopping. That share of the training data (stratified) is held out
    before augmentation, and the weights with the lowest validation loss are kept."""

    def __init__(self, hidden=16, epochs=150, lr=0.01, l2=1e-3, dropout=0.2,
                 n_augment=0, pooling="last", val_fraction=0.0, seed=0):
        self.hidden, self.epochs, self.lr, self.l2 = hidden, epochs, lr, l2
        self.dropout, self.n_augment, self.seed = dropout, n_augment, seed
        self.pooling, self.val_fraction = pooling, val_fraction

    def _penalty(self):
        return 0.5 * self.l2 * sum((p ** 2).sum() for name, p in self.net_.named_parameters()
                                   if "weight" in name)

    def fit(self, X, y):
        X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
        rng = np.random.default_rng(self.seed)
        torch.manual_seed(self.seed)
        val = np.zeros(len(y), dtype=bool)
        if self.val_fraction > 0:
            for label in (0, 1):
                members = rng.permutation(np.where(y == label)[0])
                val[members[:max(1, int(round(self.val_fraction * len(members))))]] = True
        self.mean_ = X[~val].reshape(-1, X.shape[2]).mean(axis=0)
        self.std_ = X[~val].reshape(-1, X.shape[2]).std(axis=0) + 1e-8
        Z = (X - self.mean_) / self.std_
        Zv = torch.tensor(Z[val], dtype=torch.float32)
        yv = torch.tensor(y[val], dtype=torch.float32)
        Za, ya = augment_sequences(Z[~val], y[~val], n_copies=self.n_augment, seed=self.seed)
        Za, ya = torch.tensor(Za, dtype=torch.float32), torch.tensor(ya, dtype=torch.float32)

        self.net_ = LSTMNet(Z.shape[2], self.hidden, self.dropout, self.pooling)
        optimizer = torch.optim.Adam(self.net_.parameters(), lr=self.lr)
        bce = nn.BCEWithLogitsLoss()
        self.loss_curve_, self.val_curve_ = [], []
        best, best_loss = None, np.inf
        for epoch in range(1, self.epochs + 1):
            self.net_.train()
            optimizer.zero_grad()
            loss = bce(self.net_(Za), ya) + self._penalty()
            loss.backward()
            optimizer.step()
            self.loss_curve_.append(loss.item())
            if len(yv):
                self.net_.eval()
                with torch.no_grad():
                    val_loss = bce(self.net_(Zv), yv).item()
                self.val_curve_.append(val_loss)
                if val_loss < best_loss:
                    best_loss, best, self.best_epoch_ = val_loss, copy.deepcopy(self.net_.state_dict()), epoch
        if best is not None:
            self.net_.load_state_dict(best)
        self.net_.eval()
        return self

    def predict_proba(self, X):
        Z = (np.asarray(X, dtype=float) - self.mean_) / self.std_
        with torch.no_grad():
            p = torch.sigmoid(self.net_(torch.tensor(Z, dtype=torch.float32))).numpy().astype(float)
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
