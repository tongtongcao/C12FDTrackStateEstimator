import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
import numpy as np


# -----------------------------
# Dataset
class TrackDataset(Dataset):
    def __init__(self, hits_list, states, normalize=True, hit_stats=None, state_stats=None):
        assert len(hits_list) == len(states), "hits_list 和 states 数量必须相同"
        self.hits_list = hits_list
        self.states = np.array(states, dtype=np.float32)
        self.normalize = normalize

        # 如果用户没传统计量，就默认 1.0（相当于不归一化）
        self.hit_stats = hit_stats or {
            "doca_mean": 0.0, "doca_std": 1.0,
            "xm_mean": 0.0, "xm_std": 1.0,
            "xr_mean": 0.0, "xr_std": 1.0,
            "yr_mean": 0.0, "yr_std": 1.0,
            "z_mean": 0.0, "z_std": 1.0,
        }
        self.state_stats = state_stats or {k: (0.0, 1.0) for k in ["x", "y", "tx", "ty", "Q"]}

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        hits = self.hits_list[idx].astype(np.float32)
        state = self.states[idx].copy()

        if self.normalize:
            hits[:, 0] = (hits[:, 0] - self.hit_stats["doca_mean"]) / self.hit_stats["doca_std"]
            hits[:, 1] = (hits[:, 1] - self.hit_stats["xm_mean"]) / self.hit_stats["xm_std"]
            hits[:, 2] = (hits[:, 2] - self.hit_stats["xr_mean"]) / self.hit_stats["xr_std"]
            hits[:, 3] = (hits[:, 3] - self.hit_stats["yr_mean"]) / self.hit_stats["yr_std"]
            hits[:, 4] = (hits[:, 4] - self.hit_stats["z_mean"]) / self.hit_stats["z_std"]
            for i, key in enumerate(["x", "y", "tx", "ty", "Q"]):
                mean, std = self.state_stats[key]
                state[i] = (state[i] - mean) / std

        return torch.tensor(hits, dtype=torch.float32), torch.tensor(state, dtype=torch.float32)

    def denormalize_state(self, normed_state):
        s = normed_state.clone().detach().cpu().numpy()
        for i, key in enumerate(["x", "y", "tx", "ty", "Q"]):
            mean, std = self.state_stats[key]
            s[..., i] = s[..., i] * std + mean
        return s

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

    padded_hits = torch.stack(padded_hits)  # [B, max_len, 5]
    mask = torch.tensor(mask, dtype=torch.bool)  # [B, max_len]
    states = torch.stack(states)  # [B, 5]
    return padded_hits, states, mask


# -----------------------------
# Transformer Model with padding mask
class TrackTransformer(pl.LightningModule):
    def __init__(self, input_dim=5, hidden_dim=32, nhead=4, num_layers=2, lr=1e-3):
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
            nn.Linear(hidden_dim, 5)
        )

        self.loss_fn = nn.MSELoss()

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        """
        x: [B, max_len, 5] (doca, xm, xr, yr, z)
        mask: [B, max_len], True = padding
        """
        # ---- 保证 mask 总是 tensor ----
        if mask is None:
            mask = torch.zeros(x.shape[0], x.shape[1], dtype=torch.bool, device=x.device)

        # embedding
        x_emb = self.embedding(x)  # [B, max_len, hidden_dim]

        # transformer
        x_trans = self.transformer(x_emb, src_key_padding_mask=mask)

        # 平均池化排除 padding
        valid_mask = ~mask  # True = valid
        lengths = valid_mask.sum(dim=1, keepdim=True)  # [B, 1]
        x_trans = x_trans * valid_mask.unsqueeze(-1)
        x_pooled = x_trans.sum(dim=1) / lengths

        out = self.fc(x_pooled)
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
        x: [B, N, 5]  输入的 hits 数据
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
