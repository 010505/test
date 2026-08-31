from __future__ import annotations

import torch
from torch import nn

from .backbones import EXPERIMENTAL_MODEL_NAMES, build_experimental_model
from .topology import normalized_adjacency


class STGCNBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, dropout: float = 0.1):
        super().__init__()
        self.temporal = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, (9, 1), stride=(stride, 1), padding=(4, 0), bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout),
        )
        if in_channels == out_channels and stride == 1:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=(stride, 1), bias=False),
                nn.BatchNorm2d(out_channels),
            )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        # Aggregate each node with its anatomically connected neighbours.
        spatial = torch.einsum("nctv,vw->nctw", x, adjacency)
        return self.activation(self.temporal(spatial) + self.residual(x))


class HandSTGCN(nn.Module):
    def __init__(self, num_classes: int, dropout: float = 0.15, use_graph: bool = True, single_frame: bool = False):
        super().__init__()
        adjacency = normalized_adjacency() if use_graph else torch.eye(22).numpy().astype("float32")
        self.register_buffer("adjacency", torch.from_numpy(adjacency))
        self.single_frame = single_frame
        self.input_norm = nn.BatchNorm1d(3 * 22)
        self.blocks = nn.ModuleList([
            STGCNBlock(3, 32, dropout=dropout),
            STGCNBlock(32, 64, stride=2, dropout=dropout),
            STGCNBlock(64, 96, stride=2, dropout=dropout),
            STGCNBlock(96, 128, dropout=dropout),
        ])
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != 3 or x.shape[3] != 22:
            raise ValueError("input must have shape [N, 3, T, 22]")
        if self.single_frame:
            middle = x.shape[2] // 2
            x = x[:, :, middle:middle + 1, :].expand(-1, -1, x.shape[2], -1)
        n, c, t, v = x.shape
        x = x.permute(0, 1, 3, 2).reshape(n, c * v, t)
        x = self.input_norm(x)
        x = x.reshape(n, c, v, t).permute(0, 1, 3, 2)
        for block in self.blocks:
            x = block(x, self.adjacency)
        return self.classifier(x.mean(dim=(2, 3)))


class FlatMLP(nn.Module):
    """Non-graph baseline over the complete resampled coordinate sequence."""
    def __init__(self, num_classes: int, frames: int = 64, dropout: float = .25):
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * frames * 22, 512), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(512, 128), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class FocalLoss(nn.Module):
    """Focal loss to down-weight easy samples and focus on hard confusable pairs.

    The modulating factor (1-p_t)^gamma suppresses the loss on already-correct
    predictions, keeping gradient on the misclassified tail (e.g. swipe_x / swipe_v
    / swipe_plus). An optional per-class weight vector may be supplied via `alpha`.
    """

    def __init__(self, num_classes: int, gamma: float = 2.0, alpha: float | None = None, label_smoothing: float = 0.0):
        super().__init__()
        if gamma < 0:
            raise ValueError("gamma must be non-negative")
        self.gamma = gamma
        self.alpha = alpha
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = torch.log_softmax(logits, dim=1)
        gathered = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        pt = torch.exp(gathered)
        loss = -(1 - pt).pow(self.gamma) * gathered
        if self.alpha is not None:
            alpha = torch.as_tensor(self.alpha, dtype=logits.dtype, device=logits.device)
            loss = alpha.gather(0, targets.long()) * loss
        if self.label_smoothing > 0:
            loss = loss + self.label_smoothing / logits.shape[1] * log_probs.sum(1)
        return loss.mean()


def build_model(
    name: str,
    num_classes: int,
    frames: int = 64,
    dropout: float = .15,
    ablation: str = "none",
    model_config: dict | None = None,
):
    if name == "mlp":
        return FlatMLP(num_classes, frames, dropout)
    if name == "stgcn":
        return HandSTGCN(num_classes, dropout, use_graph=ablation != "no_graph", single_frame=ablation == "single_frame")
    if name in EXPERIMENTAL_MODEL_NAMES:
        return build_experimental_model(name, num_classes, dropout, model_config)
    raise ValueError(f"unknown model: {name}")
