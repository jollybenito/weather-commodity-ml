"""Deep Learning: Long Short-Term Memory (LSTM) Sequence Model for Commodity Returns."""

from typing import Optional
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler


class LSTMNetwork(nn.Module):
    """Multi-layer LSTM network with linear projection head."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 32,
        num_layers: int = 2,
        dropout: float = 0.2
    ):
        super(LSTMNetwork, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch_size, seq_len, input_dim]
        lstm_out, _ = self.lstm(x)
        # Take hidden state at the last time step
        last_hidden = lstm_out[:, -1, :]
        out = self.fc(last_hidden)
        return out.squeeze(-1)


class LSTMRegressor:
    """Scikit-learn compatible wrapper for LSTM sequence modeling."""

    def __init__(
        self,
        seq_len: int = 15,
        hidden_dim: int = 32,
        num_layers: int = 2,
        epochs: int = 25,
        batch_size: int = 64,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        device: Optional[str] = None
    ):
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.scaler = StandardScaler()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[LSTMNetwork] = None

    def _create_sequences(
        self,
        X_scaled: np.ndarray,
        y: Optional[np.ndarray] = None
    ):
        num_samples = len(X_scaled) - self.seq_len + 1
        num_features = X_scaled.shape[1]

        seqs = np.zeros((num_samples, self.seq_len, num_features), dtype=np.float32)
        for i in range(num_samples):
            seqs[i] = X_scaled[i:i + self.seq_len]

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
        self.model = LSTMNetwork(
            input_dim=num_features,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers
        ).to(self.device)
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

        pad = np.zeros(self.seq_len - 1)
        return np.concatenate([pad, preds])

