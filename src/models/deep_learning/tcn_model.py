"""Deep Learning: Temporal Convolutional Network (TCN) for Commodity Return Forecasting.

Implements causal dilated 1D convolutions with residual blocks.
Causal padding guarantees strictly zero lookahead information leakage.
Dilations (1, 2, 4, 8) expand the receptive field across multi-week horizons.
"""

from typing import List, Optional
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler


class Chomp1d(nn.Module):
    """Trims trailing right padding to enforce causality in 1D convolutions."""
    def __init__(self, chomp_size: int):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """Single residual block of dilated causal convolutions."""
    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        kernel_size: int,
        stride: int,
        dilation: int,
        padding: int,
        dropout: float = 0.2
    ):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(
            n_inputs, n_outputs, kernel_size,
            stride=stride, padding=padding, dilation=dilation
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(
            n_outputs, n_outputs, kernel_size,
            stride=stride, padding=padding, dilation=dilation
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.dropout1,
            self.conv2, self.chomp2, self.relu2, self.dropout2
        )
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    """Multi-layer dilated Temporal Convolutional Network backbone."""
    def __init__(
        self,
        num_inputs: int,
        num_channels: List[int],
        kernel_size: int = 3,
        dropout: float = 0.2
    ):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            padding = (kernel_size - 1) * dilation_size
            layers.append(
                TemporalBlock(
                    in_channels, out_channels, kernel_size,
                    stride=1, dilation=dilation_size, padding=padding, dropout=dropout
                )
            )
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class TCNModule(nn.Module):
    """Full TCN network with projection head for scalar return prediction."""
    def __init__(
        self,
        num_features: int,
        num_channels: List[int] = [32, 32, 16],
        kernel_size: int = 3,
        dropout: float = 0.2
    ):
        super(TCNModule, self).__init__()
        self.tcn = TemporalConvNet(num_features, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch_size, num_features, seq_len]
        y = self.tcn(x)
        # Take the last time step output
        out = self.linear(y[:, :, -1])
        return out.squeeze(-1)


class TCNRegressor:
    """Scikit-learn compatible wrapper for TCN sequence modeling."""

    def __init__(
        self,
        seq_len: int = 15,
        epochs: int = 25,
        batch_size: int = 64,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        device: Optional[str] = None
    ):
        self.seq_len = seq_len
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.scaler = StandardScaler()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[TCNModule] = None

    def _create_sequences(
        self,
        X_scaled: np.ndarray,
        y: Optional[np.ndarray] = None
    ):
        num_samples = len(X_scaled) - self.seq_len + 1
        num_features = X_scaled.shape[1]
        
        seqs = np.zeros((num_samples, num_features, self.seq_len), dtype=np.float32)
        for i in range(num_samples):
            # Transpose to [num_features, seq_len] for Conv1d
            seqs[i] = X_scaled[i:i + self.seq_len].T

        if y is not None:
            targets = y[self.seq_len - 1:].astype(np.float32)
            return seqs, targets
        return seqs

    def fit(self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray):
        X_arr = X.values if isinstance(X, pd.DataFrame) else X
        y_arr = y.values if isinstance(y, pd.Series) else y

        X_scaled = self.scaler.fit_transform(X_arr)
        seqs, targets = self._create_sequences(X_scaled, y_arr)

        dataset = TensorDataset(torch.tensor(seqs), torch.tensor(targets))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        num_features = X_scaled.shape[1]
        self.model = TCNModule(num_features=num_features).to(self.device)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        criterion = nn.HuberLoss(delta=0.02)

        self.model.train()
        for epoch in range(self.epochs):
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                preds = self.model(batch_x)
                loss = criterion(preds, batch_y)
                loss.backward()
                optimizer.step()

        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model is not fitted yet.")

        X_arr = X.values if isinstance(X, pd.DataFrame) else X
        X_scaled = self.scaler.transform(X_arr)
        seqs = self._create_sequences(X_scaled)

        self.model.eval()
        with torch.no_grad():
            tensor_seqs = torch.tensor(seqs).to(self.device)
            preds = self.model(tensor_seqs).cpu().numpy()

        # Prepend zeros/nans for initial seq_len - 1 rows to align with original length
        pad = np.zeros(self.seq_len - 1)
        return np.concatenate([pad, preds])

