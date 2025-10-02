import time
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

class MLPRegressorTorch:
    """
    A tiny MLP regressor implemented in PyTorch.

    Reasoning:
    - Provides a simple deep learning baseline.
    - Keeps parameters small for fast runs on CPU or MPS.
    - Consistent .fit/.predict API for drop-in benchmarking.
    """
    def __init__(self, in_dim, hidden=(64, 64), lr=1e-3, epochs=300, device=None):
        self.epochs = int(epochs)
        self.lr = float(lr)
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        layers = []
        h = [in_dim] + list(hidden) + [1]
        for i in range(len(h) - 1):
            layers.append(nn.Linear(h[i], h[i+1]))
            if i < len(h) - 2:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers).to(self.device)
        self.train_time_ = None

    def fit(self, X, y):
        X = torch.tensor(X, dtype=torch.float32, device=self.device)
        y = torch.tensor(y, dtype=torch.float32, device=self.device).view(-1, 1)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()
        t0 = time.time()
        self.net.train()
        for _ in range(self.epochs):
            opt.zero_grad()
            pred = self.net(X)
            loss = loss_fn(pred, y)
            loss.backward()
            opt.step()
        self.train_time_ = time.time() - t0
        return self

    def predict(self, X):
        self.net.eval()
        X = torch.tensor(X, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            pred = self.net(X).cpu().numpy().ravel()
        return pred


class MLPDropoutRegressorTorch:
    """MLP variant with dropout used for Monte Carlo uncertainty estimates."""

    def __init__(
        self,
        in_dim: int,
        hidden: tuple[int, ...] = (128, 128),
        dropout: float = 0.1,
        lr: float = 1e-3,
        epochs: int = 400,
        device: Optional[str] = None,
        mc_samples: int = 30,
    ) -> None:
        if dropout <= 0 or dropout >= 1:
            raise ValueError("Dropout must be in (0, 1).")
        self.epochs = int(epochs)
        self.lr = float(lr)
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.dropout = float(dropout)
        self.mc_samples = int(mc_samples)

        layers = []
        dims = [in_dim] + list(hidden) + [1]
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(p=self.dropout))
        self.net = nn.Sequential(*layers).to(self.device)
        self.train_time_ = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MLPDropoutRegressorTorch":
        X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device)
        y_tensor = torch.tensor(y, dtype=torch.float32, device=self.device).view(-1, 1)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()
        t0 = time.time()
        self.net.train()
        for _ in range(self.epochs):
            opt.zero_grad()
            pred = self.net(X_tensor)
            loss = loss_fn(pred, y_tensor)
            loss.backward()
            opt.step()
        self.train_time_ = time.time() - t0
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        mean, _ = self.predict_with_uncertainty(X)
        return mean

    def predict_with_uncertainty(
        self, X: np.ndarray, mc_samples: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        samples = int(mc_samples or self.mc_samples)
        if samples <= 0:
            raise ValueError("mc_samples must be positive.")

        was_training = self.net.training
        self.net.train()  # keep dropout active
        X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device)
        preds = []
        with torch.no_grad():
            for _ in range(samples):
                preds.append(self.net(X_tensor).cpu().numpy().ravel())
        pred_array = np.stack(preds, axis=0)
        mean = pred_array.mean(axis=0)
        std = pred_array.std(axis=0)
        if not was_training:
            self.net.eval()
        return mean, std
