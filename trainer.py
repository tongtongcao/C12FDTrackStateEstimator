import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
import numpy as np


# -----------------------------
# Dataset
class TrackDataset(Dataset):
    def __init__(self, hits_list, states):
        """
        hits_list : list of np.ndarray, shape [num_hits, 3] (doca, doca_err, z)
        states    : np.ndarray, shape [N, 7]
        """
        assert len(hits_list) == len(states), "hits_list 和 states 数量必须相同"
        self.hits_list = hits_list
        self.states = np.array(states, dtype=np.float32)

    def __len__(self):
        return len(self.hits_list)

    def __getitem__(self, idx):
        hits = torch.tensor(self.hits_list[idx], dtype=torch.float32)
        state = torch.tensor(self.states[idx], dtype=torch.float32)
        return hits, state


# -----------------------------
# Collate function for variable-length hits
def collate_fn(batch):
    hits, states = zip(*batch)
    max_len = max(h.shape[0] for h in hits)

    padded_hits = []
    mask = []
    for h in hits:
        orig_len = h.shape[0]
        pad_len = max_len - orig_len
        if pad_len > 0:
            pad = torch.zeros(pad_len, h.shape[1])
            h = torch.cat([h, pad], dim=0)
        padded_hits.append(h)
        # mask: True = padding, False = valid
        mask.append([False]*orig_len + [True]*pad_len)

    padded_hits = torch.stack(padded_hits)  # [B, max_len, 3]
    mask = torch.tensor(mask, dtype=torch.bool)  # [B, max_len]
    states = torch.stack(states)  # [B, 7]
    return padded_hits, states, mask


# -----------------------------
# Transformer Model with doca_error weighting and padding mask
class TrackTransformerWithError(pl.LightningModule):
    def __init__(self, input_dim=3, hidden_dim=32, nhead=4, num_layers=2, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr

        self.embedding = nn.Linear(input_dim, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 4,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 7)
        )

        self.loss_fn = nn.MSELoss()

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        """
        x: [B, max_len, 3] (doca, doca_err, z)
        mask: [B, max_len], True = padding
        """
        # ---- 保证 mask 总是 tensor ----
        if mask is None:
            mask = torch.zeros(x.shape[0], x.shape[1], dtype=torch.bool, device=x.device)

        # embedding
        x_emb = self.embedding(x)  # [B, max_len, hidden_dim]

        # transformer
        x_trans = self.transformer(x_emb, src_key_padding_mask=mask)

        # weighting by doca_error
        doca_error = x[:, :, 1:2]  # [B, max_len, 1]
        weights = 1.0 / (torch.clamp(doca_error, min=1e-3) ** 2)

        # zero out padded positions
        weights = weights * (~mask.unsqueeze(-1))

        # weighted average
        x_weighted = (x_trans * weights).sum(dim=1) / (weights.sum(dim=1) + 1e-6)

        out = self.fc(x_weighted)
        return out

    def training_step(self, batch, batch_idx):
        x, y, mask = batch
        y_hat = self(x, mask)
        loss = self.loss_fn(y_hat, y)
        self.log("train_loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, mask = batch
        y_hat = self(x, mask)
        loss = self.loss_fn(y_hat, y)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

class TrackTransformerWrapper(nn.Module):
    """
    包装原始模型，使其在推理时自动生成 mask（全 False）。
    这样导出的 TorchScript 模型在 DJL 中使用时，只需输入 hits 即可。
    """
    def __init__(self, core_model: nn.Module):
        super().__init__()
        self.core = core_model

    def forward(self, x: torch.Tensor):
        """
        x: [B, N, 3]  输入的 hits 数据
        自动生成 mask: [B, N]，全 False
        """
        mask = torch.zeros(x.shape[0], x.shape[1], dtype=torch.bool, device=x.device)
        return self.core(x, mask)


# -----------------------------
# Loss Tracker Callback
class LossTracker(Callback):
    def __init__(self):
        self.train_losses = []
        self.val_losses = []

    def on_train_epoch_end(self, trainer, pl_module):
        if "train_loss" in trainer.callback_metrics:
            self.train_losses.append(trainer.callback_metrics["train_loss"].item())

    def on_validation_epoch_end(self, trainer, pl_module):
        if "val_loss" in trainer.callback_metrics:
            self.val_losses.append(trainer.callback_metrics["val_loss"].item())
