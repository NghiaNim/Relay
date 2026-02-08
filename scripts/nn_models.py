"""Neural network models for EEG intent classification."""

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

SEED = 42


def _set_seed():
    if TORCH_AVAILABLE:
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)


class EEG1DCNN(nn.Module):
    """1D CNN for EEG. Input: (batch, 1, channels, samples)."""

    def __init__(self, n_channels=6, n_samples=6000, n_classes=5, dropout=0.3):
        super().__init__()
        self.conv1 = nn.Conv1d(n_channels, 32, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(2)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.pool3 = nn.AdaptiveAvgPool1d(1)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(128, n_classes)

    def forward(self, x):
        x = x.squeeze(1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        x = x.flatten(1)
        x = self.drop(x)
        return self.fc(x)


class EEGTCN(nn.Module):
    """Temporal Convolutional Network with dilated convolutions."""

    def __init__(self, n_channels=6, n_samples=6000, n_classes=5, n_levels=4, n_filters=32, dropout=0.3):
        super().__init__()
        layers = []
        in_ch = n_channels
        for i in range(n_levels):
            out_ch = n_filters * (2 ** min(i, 2))
            pad = 2 ** i
            layers.append(
                nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=pad, dilation=2 ** i)
            )
            in_ch = out_ch
        self.convs = nn.ModuleList(layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(in_ch, n_classes)

    def forward(self, x):
        x = x.squeeze(1)
        for conv in self.convs:
            x = F.relu(conv(x))
        x = self.pool(x)
        x = x.flatten(1)
        x = self.drop(x)
        return self.fc(x)


class EEGLSTM(nn.Module):
    """LSTM for EEG. Input: (batch, 1, channels, samples) → permute to (batch, seq, channels)."""

    def __init__(self, n_channels=6, n_samples=6000, n_classes=5, hidden=64, n_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(n_channels, hidden, n_layers, batch_first=True, dropout=dropout if n_layers > 1 else 0)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden, n_classes)

    def forward(self, x):
        x = x.squeeze(1)
        x = x.permute(0, 2, 1)
        out, (h, _) = self.lstm(x)
        x = h[-1]
        x = self.drop(x)
        return self.fc(x)


class EEGTransformer(nn.Module):
    """Self-attention transformer for EEG."""

    def __init__(self, n_channels=6, n_samples=6000, n_classes=5, d_model=64, n_heads=4, n_layers=2, dropout=0.3):
        super().__init__()
        self.proj = nn.Linear(n_channels, d_model)
        enc = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=d_model * 4, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(enc, n_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(d_model, n_classes)

    def forward(self, x):
        x = x.squeeze(1)
        x = x.permute(0, 2, 1)
        x = self.proj(x)
        x = self.transformer(x)
        x = x.permute(0, 2, 1)
        x = self.pool(x).squeeze(-1)
        return self.fc(x)


class EEGMamba(nn.Module):
    """Simplified SSM-style model using causal conv + gated MLP (Mamba-inspired, no mamba-ssm dep)."""

    def __init__(self, n_channels=6, n_samples=6000, n_classes=5, d_state=16, dropout=0.3):
        super().__init__()
        self.proj = nn.Conv1d(n_channels, d_state, 1)
        self.conv = nn.Conv1d(d_state, d_state * 2, kernel_size=4, padding=4 - 1, groups=d_state)
        self.fc1 = nn.Linear(d_state * 2, d_state * 4)
        self.fc2 = nn.Linear(d_state * 4, d_state)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.drop = nn.Dropout(dropout)
        self.fc_out = nn.Linear(d_state, n_classes)

    def forward(self, x):
        x = x.squeeze(1)
        x = self.proj(x)
        x = self.conv(x)[:, :, : -3]
        x = F.gelu(x)
        x = x.permute(0, 2, 1)
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)
        x = self.pool(x.permute(0, 2, 1)).squeeze(-1)
        x = self.drop(x)
        return self.fc_out(x)


def train_torch_model(model, X, y, epochs=50, lr=1e-3, batch_size=32):
    """Train PyTorch model. X: (N, 1, C, T), y: labels (int). Returns dict with model, scaler, etc."""
    _set_seed()
    import torch
    from torch.utils.data import TensorDataset, DataLoader

    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.long)
    ds = TensorDataset(X_t, y_t)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    model.train()
    for _ in range(epochs):
        for bx, by in loader:
            opt.zero_grad()
            loss = F.cross_entropy(model(bx), by)
            loss.backward()
            opt.step()

    return {
        "model": model,
        "model_state": model.state_dict(),
        "model_class": model.__class__.__name__,
        "n_classes": len(np.unique(y)),
    }
