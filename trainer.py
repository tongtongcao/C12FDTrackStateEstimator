import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
import numpy as np


# -----------------------------
# Dataset for track-level learning
class TrackDataset(Dataset):
    """
    PyTorch Dataset for track reconstruction.

    Each sample consists of:
      - a variable-length list of hits with 5 features per hit
      - a 5D target state vector (x, y, tx, ty, Q)

    Optional normalization is applied using precomputed statistics.
    """
    def __init__(self, hits_list, states, normalize=True, hit_stats=None, state_stats=None):
        # hits_list and states must have the same number of tracks
        assert len(hits_list) == len(states), "hits_list and states must have the same length"

        self.hits_list = hits_list
        self.states = np.array(states, dtype=np.float32)
        self.normalize = normalize

        # Default hit normalization statistics (identity transform if not provided)
        self.hit_stats = hit_stats or {
            "doca_mean": 0.0, "doca_std": 1.0,
            "xm_mean": 0.0, "xm_std": 1.0,
            "xr_mean": 0.0, "xr_std": 1.0,
            "yr_mean": 0.0, "yr_std": 1.0,
            "z_mean": 0.0, "z_std": 1.0,
        }

        # Default state normalization statistics
        self.state_stats = state_stats or {k: (0.0, 1.0) for k in ["x", "y", "tx", "ty", "Q"]}

    def __len__(self):
        """Return the number of tracks."""
        return len(self.states)

    def __getitem__(self, idx):
        """
        Return one training sample.

        Returns
        -------
        hits : torch.FloatTensor, shape [num_hits, 5]
            Hit-level input features.
        state : torch.FloatTensor, shape [5]
            Target track state.
        """
        hits = self.hits_list[idx].astype(np.float32)
        state = self.states[idx].copy()

        # Apply normalization if enabled
        if self.normalize:
            hits[:, 0] = (hits[:, 0] - self.hit_stats["doca_mean"]) / self.hit_stats["doca_std"]
            hits[:, 1] = (hits[:, 1] - self.hit_stats["xm_mean"]) / self.hit_stats["xm_std"]
            hits[:, 2] = (hits[:, 2] - self.hit_stats["xr_mean"]) / self.hit_stats["xr_std"]
            hits[:, 3] = (hits[:, 3] - self.hit_stats["yr_mean"]) / self.hit_stats["yr_std"]
            hits[:, 4] = (hits[:, 4] - self.hit_stats["z_mean"]) / self.hit_stats["z_std"]

            for i, key in enumerate(["x", "y", "tx", "ty", "Q"]):
                mean, std = self.state_stats[key]
                state[i] = (state[i] - mean) / std

        return (
            torch.tensor(hits, dtype=torch.float32),
            torch.tensor(state, dtype=torch.float32)
        )

    def denormalize_state(self, normed_state):
        """
        Convert normalized state variables back to physical units.
        """
        s = normed_state.clone().detach().cpu().numpy()
        for i, key in enumerate(["x", "y", "tx", "ty", "Q"]):
            mean, std = self.state_stats[key]
            s[..., i] = s[..., i] * std + mean
        return s


# -----------------------------
# Collate function for variable-length hit sequences
def collate_fn(batch):
    """
    Custom collate function to handle variable-length hit sequences.

    Pads hit sequences to the maximum length in the batch and
    generates a padding mask.

    Returns
    -------
    padded_hits : torch.FloatTensor, shape [B, max_len, 5]
    states       : torch.FloatTensor, shape [B, 5]
    mask         : torch.BoolTensor, shape [B, max_len]
        True indicates padding positions.
    """
    hits, states = zip(*batch)
    max_len = max(h.shape[0] for h in hits)

    padded_hits = []
    mask = []

    for h in hits:
        orig_len = h.shape[0]
        pad_len = max_len - orig_len

        # Pad hits with zeros
        if pad_len > 0:
            pad = torch.zeros(pad_len, h.shape[1])
            h = torch.cat([h, pad], dim=0)

        padded_hits.append(h)

        # Mask: False = valid hit, True = padding
        mask.append([False] * orig_len + [True] * pad_len)

    padded_hits = torch.stack(padded_hits)     # [B, max_len, 5]
    states = torch.stack(states)               # [B, 5]
    mask = torch.tensor(mask, dtype=torch.bool)

    return padded_hits, states, mask


# -----------------------------
# Transformer-based model with padding mask support
class TrackTransformer(pl.LightningModule):
    """
    Transformer encoder model for track state regression.

    The model:
      - embeds hit-level features
      - applies a Transformer encoder with padding masks
      - pools over valid hits
      - predicts a 5D track state
    """
    def __init__(self, input_dim=5, hidden_dim=32, nhead=4, num_layers=2, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr

        # Linear embedding from hit features to transformer dimension
        self.embedding = nn.Linear(input_dim, hidden_dim)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 4,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Final regression head
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 5)
        )

        self.loss_fn = nn.MSELoss()

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        """
        Parameters
        ----------
        x : torch.FloatTensor, shape [B, max_len, 5]
            Hit-level input features.
        mask : torch.BoolTensor, shape [B, max_len]
            Padding mask (True = padding).
        """
        # Ensure mask is always defined
        if mask is None:
            mask = torch.zeros(x.shape[0], x.shape[1], dtype=torch.bool, device=x.device)

        # Embed hits
        x_emb = self.embedding(x)  # [B, max_len, hidden_dim]

        # Transformer encoder with padding mask
        x_trans = self.transformer(x_emb, src_key_padding_mask=mask)

        # Masked mean pooling over valid hits
        valid_mask = ~mask
        lengths = valid_mask.sum(dim=1, keepdim=True)  # [B, 1]
        x_trans = x_trans * valid_mask.unsqueeze(-1)
        x_pooled = x_trans.sum(dim=1) / lengths

        # Predict track state
        out = self.fc(x_pooled)
        return out

    def training_step(self, batch, batch_idx):
        """One training step."""
        x, y, mask = batch
        y_hat = self(x, mask)
        loss = self.loss_fn(y_hat, y)
        self.log("train_loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """One validation step."""
        x, y, mask = batch
        y_hat = self(x, mask)
        loss = self.loss_fn(y_hat, y)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)

    def configure_optimizers(self):
        """Configure optimizer."""
        return torch.optim.Adam(self.parameters(), lr=self.lr)


# -----------------------------
# Wrapper for inference without explicit mask input
class TrackTransformerWrapper(nn.Module):
    """
    Wrapper around TrackTransformer that automatically generates
    an all-False padding mask during inference.

    This allows the exported TorchScript model to be used with
    a single input tensor (hits only), e.g. in DJL.
    """
    def __init__(self, core_model: nn.Module):
        super().__init__()
        self.core = core_model

    def forward(self, x: torch.Tensor):
        """
        Parameters
        ----------
        x : torch.FloatTensor, shape [B, N, 5]
            Hit-level input features.
        """
        mask = torch.zeros(x.shape[0], x.shape[1], dtype=torch.bool, device=x.device)
        return self.core(x, mask)


# -----------------------------
# Callback for tracking training and validation losses
class LossTracker(Callback):
    """
    PyTorch Lightning callback to record train and validation losses
    at the end of each epoch.
    """
    def __init__(self):
        self.train_losses = []
        self.val_losses = []

    def on_train_epoch_end(self, trainer, pl_module):
        if "train_loss" in trainer.callback_metrics:
            self.train_losses.append(trainer.callback_metrics["train_loss"].item())

    def on_validation_epoch_end(self, trainer, pl_module):
        if "val_loss" in trainer.callback_metrics:
            self.val_losses.append(trainer.callback_metrics["val_loss"].item())
