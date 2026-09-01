from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from .topology import (
    NUM_NODES,
    binary_adjacency,
    laplacian_eigenpairs,
    normalized_adjacency,
)


EXPERIMENTAL_MODEL_NAMES = (
    "spectral_pe_stgcn",
    "spectral_pe_qkv",
    "gwnet_adaptive_support",
    "agcrn_factorized_adjacency",
    "velocity_agcrn",
    "gated_agcrn",
    "velocity_gated_agcrn",
    "spectral_pe_qkv_stable",
)


def _prepare_input(x: torch.Tensor, input_norm: nn.BatchNorm1d) -> torch.Tensor:
    if x.ndim != 4 or x.shape[3] != NUM_NODES:
        raise ValueError(f"input must have shape [N, C, T, {NUM_NODES}]")
    n, channels, frames, nodes = x.shape
    x = x.permute(0, 1, 3, 2).reshape(n, channels * nodes, frames)
    x = input_norm(x)
    return x.reshape(n, channels, nodes, frames).permute(0, 1, 3, 2)


def _velocity_channels(x: torch.Tensor) -> torch.Tensor:
    """Frame-to-frame displacement (first frame zero-padded)."""
    velocity = torch.zeros_like(x)
    velocity[:, :, 1:, :] = x[:, :, 1:, :] - x[:, :, :-1, :]
    return torch.cat([x, velocity], dim=1)


class TemporalGate(nn.Module):
    """Squeeze-and-excite over the time axis, reweighting frames per channel.

    Collapses the time dimension to a per-channel statistic, then learns a small
    gain in [0, 1] to emphasise the most informative moments of a gesture.
    """

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        squeezed = max(1, channels // reduction)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, NUM_NODES)),
            nn.Conv2d(channels, squeezed, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(squeezed, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gate(x)


class AdjacencyAggregate(nn.Module):
    """Fixed A aggregation used by the spectral-PE ST-GCN experiment."""

    def forward(
        self,
        x: torch.Tensor,
        adjacency: torch.Tensor | None = None,
        node_embeddings: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del node_embeddings
        if adjacency is None:
            raise ValueError("adjacency is required")
        return torch.einsum("nctv,vw->nctw", x, adjacency)


class GraphWaveNetSupportProjection(nn.Module):
    """Graph WaveNet-style diffusion over fixed and adaptive supports.

    For two supports and order two, concatenate
    [X, A_phys X, A_phys^2 X, A_adp X, A_adp^2 X] and fuse them with
    a learned 1x1 projection, matching the support treatment in Graph WaveNet.
    """

    def __init__(self, channels: int, order: int = 2):
        super().__init__()
        if order < 1:
            raise ValueError("diffusion order must be positive")
        self.order = order
        self.projection = nn.Conv2d(channels * (1 + 2 * order), channels, 1, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        adjacency: tuple[torch.Tensor, torch.Tensor] | None = None,
        node_embeddings: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del node_embeddings
        if adjacency is None or len(adjacency) != 2:
            raise ValueError("fixed and adaptive supports are required")
        outputs = [x]
        for support in adjacency:
            propagated = x
            for _ in range(self.order):
                propagated = torch.einsum("nctv,vw->nctw", propagated, support)
                outputs.append(propagated)
        return self.projection(torch.cat(outputs, dim=1))


class NodeAdaptiveGraphWaveNetSupportProjection(nn.Module):
    """Graph WaveNet diffusion with AGCRN-style node-specific weights.

    The fixed and learned supports first form
    [X, A_phys X, A_phys^2 X, A_adp X, A_adp^2 X].  Each node embedding then
    combines a shared weight pool into its own transformation W_v.
    """

    def __init__(self, channels: int, embedding_dim: int, order: int = 2):
        super().__init__()
        if order < 1:
            raise ValueError("diffusion order must be positive")
        self.order = order
        diffusion_channels = channels * (1 + 2 * order)
        self.weights_pool = nn.Parameter(torch.empty(embedding_dim, diffusion_channels, channels))
        self.bias_pool = nn.Parameter(torch.zeros(embedding_dim, channels))
        nn.init.xavier_uniform_(self.weights_pool)

    def forward(
        self,
        x: torch.Tensor,
        adjacency: tuple[torch.Tensor, torch.Tensor] | None = None,
        node_embeddings: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if adjacency is None or len(adjacency) != 2 or node_embeddings is None:
            raise ValueError("fixed/adaptive supports and node_embeddings are required")
        outputs = [x]
        for support in adjacency:
            propagated = x
            for _ in range(self.order):
                propagated = torch.einsum("nctv,vw->nctw", propagated, support)
                outputs.append(propagated)
        diffusion_features = torch.cat(outputs, dim=1)
        weights = torch.einsum("vd,dio->vio", node_embeddings, self.weights_pool)
        bias = torch.einsum("vd,do->vo", node_embeddings, self.bias_pool)
        output = torch.einsum("nitv,vio->notv", diffusion_features, weights)
        return output + bias.transpose(0, 1).unsqueeze(0).unsqueeze(2)


class EigenvalueWeightedSpatial(nn.Module):
    """Apply Z = X + (SE diag(alpha)) W_pe before a spatial operator."""

    def __init__(
        self,
        channels: int,
        eigenvalues: torch.Tensor,
        positional_encoding: torch.Tensor,
        spatial: nn.Module,
    ):
        super().__init__()
        self.register_buffer("spectral_eigenvalues", eigenvalues.clone())
        self.register_buffer("spectral_encoding", positional_encoding.clone())
        self.position_projection = nn.Linear(positional_encoding.shape[1], channels, bias=False)
        self.spatial = spatial

    def forward(
        self,
        x: torch.Tensor,
        adjacency: torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None = None,
        node_embeddings: torch.Tensor | None = None,
    ) -> torch.Tensor:
        channels = x.shape[1]
        nodes = x.shape[3]
        weighted_encoding = self.spectral_encoding * self.spectral_eigenvalues.unsqueeze(0)
        position = self.position_projection(weighted_encoding).transpose(0, 1).view(1, channels, 1, nodes)
        return self.spatial(x + position, adjacency, node_embeddings)


class MultiHeadQKVSpatial(nn.Module):
    """Full-joint per-frame multi-head Q/K/V attention.

    Every frame attends over all 22 joints. The physical hand graph is only a
    learnable soft bias: it favours anatomical edges at initialisation but never
    masks latent non-physical connections. No temporal position is encoded;
    temporal modelling remains the responsibility of the following TCN.
    """

    def __init__(
        self,
        channels: int,
        heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        if heads < 1:
            raise ValueError("heads must be positive")
        self.heads = heads
        self.head_dim = max(8, math.ceil(channels / heads))
        self.register_buffer("physical_adjacency", torch.from_numpy(binary_adjacency(include_self=True)))
        projection_width = heads * self.head_dim
        self.query_projection = nn.Linear(channels, projection_width, bias=False)
        self.key_projection = nn.Linear(channels, projection_width, bias=False)
        self.value_projection = nn.Linear(channels, projection_width, bias=False)
        self.output_projection = nn.Linear(heads * self.head_dim, channels, bias=False)
        self.attention_dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(dropout)
        self.physical_bias = nn.Parameter(torch.full((heads,), 0.5))
        self.residual_scale = nn.Parameter(torch.tensor(0.1))

    def forward(
        self,
        x: torch.Tensor,
        adjacency: torch.Tensor | None = None,
        node_embeddings: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del adjacency, node_embeddings
        batch, channels, frames, nodes = x.shape
        features = x.permute(0, 2, 3, 1)

        def project(layer: nn.Linear) -> torch.Tensor:
            projected = layer(features).view(batch, frames, nodes, self.heads, self.head_dim)
            return projected.permute(0, 1, 3, 2, 4)

        query = project(self.query_projection)
        key = project(self.key_projection)
        value = project(self.value_projection)
        scores = torch.einsum("bthid,bthjd->bthij", query, key) / math.sqrt(self.head_dim)
        soft_prior = self.physical_bias.view(1, 1, self.heads, 1, 1)
        scores = scores + soft_prior * self.physical_adjacency.view(1, 1, 1, nodes, nodes)
        attention = self.attention_dropout(scores.softmax(dim=-1))
        attended = torch.einsum("bthij,bthjd->bthid", attention, value)
        attended = attended.permute(0, 1, 3, 2, 4).reshape(batch, frames, nodes, self.heads * self.head_dim)
        output = self.output_projection(attended).permute(0, 3, 1, 2).contiguous()
        return x + self.residual_scale * self.output_dropout(output)


class StableMultiHeadQKVSpatial(nn.Module):
    """Injection-stabilised full-joint Q/K/V attention.

    Resolves the seed sensitivity of :class:`MultiHeadQKVSpatial` by adding three
    stabilisers that keep the attention distribution near-uniform at initialisation:
      * LayerNorm before the Q/K/V projection (bounds the score scale),
      * a fixed temperature 1/sqrt(head_dim) instead of a learned shared residual
        gain (the original started at 0.1, which kept the residual tiny and let the
        network latch onto random initial attention),
      * a per-head learnable temperature that can widen the distribution when a
        head is under-determined, plus a residual gain initialised to 1.0.

    The physical adjacency is still only a soft learnable bias, never a hard mask.
    """

    def __init__(self, channels: int, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        if heads < 1:
            raise ValueError("heads must be positive")
        self.heads = heads
        self.head_dim = max(8, math.ceil(channels / heads))
        self.register_buffer("physical_adjacency", torch.from_numpy(binary_adjacency(include_self=True)))
        projection_width = heads * self.head_dim
        self.input_norm = nn.LayerNorm(channels)
        self.query_projection = nn.Linear(channels, projection_width, bias=False)
        self.key_projection = nn.Linear(channels, projection_width, bias=False)
        self.value_projection = nn.Linear(channels, projection_width, bias=False)
        self.output_projection = nn.Linear(heads * self.head_dim, channels, bias=False)
        self.attention_dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(dropout)
        self.per_head_temperature = nn.Parameter(torch.full((heads,), 1.0))
        self.physical_bias = nn.Parameter(torch.full((heads,), 0.5))
        self.residual_gain = nn.Parameter(torch.tensor(1.0))

    def forward(
        self,
        x: torch.Tensor,
        adjacency: torch.Tensor | None = None,
        node_embeddings: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del adjacency, node_embeddings
        batch, channels, frames, nodes = x.shape
        features = x.permute(0, 2, 3, 1)
        features = self.input_norm(features)

        def project(layer: nn.Linear) -> torch.Tensor:
            projected = layer(features).view(batch, frames, nodes, self.heads, self.head_dim)
            return projected.permute(0, 1, 3, 2, 4)

        query = project(self.query_projection)
        key = project(self.key_projection)
        value = project(self.value_projection)
        temperature = self.per_head_temperature.clamp(min=1e-2).view(1, 1, self.heads, 1, 1)
        scores = torch.einsum("bthid,bthjd->bthij", query, key) / math.sqrt(self.head_dim) / temperature
        soft_prior = self.physical_bias.view(1, 1, self.heads, 1, 1)
        scores = scores + soft_prior * self.physical_adjacency.view(1, 1, 1, nodes, nodes)
        attention = self.attention_dropout(scores.softmax(dim=-1))
        attended = torch.einsum("bthij,bthjd->bthid", attention, value)
        attended = attended.permute(0, 1, 3, 2, 4).reshape(batch, frames, nodes, self.heads * self.head_dim)
        output = self.output_projection(attended).permute(0, 3, 1, 2).contiguous()
        return x + self.residual_gain * self.output_dropout(output)


class SpatialTemporalBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, spatial: nn.Module, stride: int = 1, dropout: float = 0.1):
        super().__init__()
        self.spatial = spatial
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

    def forward(
        self,
        x: torch.Tensor,
        adjacency: torch.Tensor | None = None,
        node_embeddings: torch.Tensor | None = None,
    ) -> torch.Tensor:
        spatial = self.spatial(x, adjacency, node_embeddings)
        return self.activation(self.temporal(spatial) + self.residual(x))


class GraphWaveNetAdaptiveAdjacency(nn.Module):
    """Unmasked full adaptive support from the original Graph WaveNet recipe."""

    def __init__(self, embedding_dim: int = 10):
        super().__init__()
        if embedding_dim < 1:
            raise ValueError("embedding_dim must be positive")
        self.nodevec1 = nn.Parameter(torch.empty(NUM_NODES, embedding_dim))
        self.nodevec2 = nn.Parameter(torch.empty(NUM_NODES, embedding_dim))
        self.register_buffer("physical_adjacency", torch.from_numpy(normalized_adjacency()))
        nn.init.xavier_uniform_(self.nodevec1)
        nn.init.xavier_uniform_(self.nodevec2)

    def adaptive_adjacency(self) -> torch.Tensor:
        scores = F.relu(self.nodevec1 @ self.nodevec2.transpose(0, 1))
        return scores.softmax(dim=1)

    def components(self) -> dict[str, torch.Tensor]:
        adaptive = self.adaptive_adjacency()
        return {
            "physical": self.physical_adjacency.detach().cpu(),
            "adaptive_support": adaptive.detach().cpu(),
        }


def _channel_pairs() -> list[tuple[int, int, int]]:
    return [(3, 32, 1), (32, 64, 2), (64, 96, 2), (96, 128, 1)]


class SpectralPESTGCN(nn.Module):
    """Experiment 1: ST-GCN with eigenvalue-weighted PE before every GCN."""

    def __init__(self, num_classes: int, dropout: float = 0.15, pe_dim: int = 8):
        super().__init__()
        eigenvalues, eigenvectors = laplacian_eigenpairs(pe_dim)
        eigenvalues = torch.from_numpy(eigenvalues)
        encoding = torch.from_numpy(eigenvectors)
        self.register_buffer("adjacency", torch.from_numpy(normalized_adjacency()))
        self.input_norm = nn.BatchNorm1d(3 * NUM_NODES)
        self.blocks = nn.ModuleList([
            SpatialTemporalBlock(
                in_channels,
                out_channels,
                EigenvalueWeightedSpatial(in_channels, eigenvalues, encoding, AdjacencyAggregate()),
                stride,
                dropout,
            )
            for in_channels, out_channels, stride in _channel_pairs()
        ])
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _prepare_input(x, self.input_norm)
        for block in self.blocks:
            x = block(x, self.adjacency)
        return self.classifier(x.mean(dim=(2, 3)))


class SpectralPEQKV(nn.Module):
    """Experiment 2: full-joint multi-head Q/K/V spatial attention + TCN."""

    def __init__(self, num_classes: int, dropout: float = 0.15, pe_dim: int = 8, heads: int = 4):
        super().__init__()
        eigenvalues, eigenvectors = laplacian_eigenpairs(pe_dim)
        eigenvalues = torch.from_numpy(eigenvalues)
        encoding = torch.from_numpy(eigenvectors)
        self.input_norm = nn.BatchNorm1d(3 * NUM_NODES)
        self.blocks = nn.ModuleList([
            SpatialTemporalBlock(
                in_channels,
                out_channels,
                EigenvalueWeightedSpatial(
                    in_channels,
                    eigenvalues,
                    encoding,
                    MultiHeadQKVSpatial(in_channels, heads=heads, dropout=dropout),
                ),
                stride,
                dropout,
            )
            for in_channels, out_channels, stride in _channel_pairs()
        ])
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _prepare_input(x, self.input_norm)
        for block in self.blocks:
            x = block(x)
        return self.classifier(x.mean(dim=(2, 3)))


class SpectralPEQKVStable(nn.Module):
    """Stabilised variant of experiment 2 (see :class:`StableMultiHeadQKVSpatial`)."""

    def __init__(self, num_classes: int, dropout: float = 0.15, pe_dim: int = 8, heads: int = 4):
        super().__init__()
        eigenvalues, eigenvectors = laplacian_eigenpairs(pe_dim)
        eigenvalues = torch.from_numpy(eigenvalues)
        encoding = torch.from_numpy(eigenvectors)
        self.input_norm = nn.BatchNorm1d(3 * NUM_NODES)
        self.blocks = nn.ModuleList([
            SpatialTemporalBlock(
                in_channels,
                out_channels,
                EigenvalueWeightedSpatial(
                    in_channels,
                    eigenvalues,
                    encoding,
                    StableMultiHeadQKVSpatial(in_channels, heads=heads, dropout=dropout),
                ),
                stride,
                dropout,
            )
            for in_channels, out_channels, stride in _channel_pairs()
        ])
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _prepare_input(x, self.input_norm)
        for block in self.blocks:
            x = block(x)
        return self.classifier(x.mean(dim=(2, 3)))


class GraphWaveNetAdaptiveSTGCN(nn.Module):
    """Experiment 3: spectral PE plus fixed/full-adaptive GWN supports."""

    def __init__(self, num_classes: int, dropout: float = 0.15, adaptive_dim: int = 10, pe_dim: int = 8):
        super().__init__()
        eigenvalues, eigenvectors = laplacian_eigenpairs(pe_dim)
        eigenvalues = torch.from_numpy(eigenvalues)
        encoding = torch.from_numpy(eigenvectors)
        self.input_norm = nn.BatchNorm1d(3 * NUM_NODES)
        self.adaptive_graph = GraphWaveNetAdaptiveAdjacency(adaptive_dim)
        self.blocks = nn.ModuleList([
            SpatialTemporalBlock(
                in_channels,
                out_channels,
                EigenvalueWeightedSpatial(
                    in_channels,
                    eigenvalues,
                    encoding,
                    GraphWaveNetSupportProjection(in_channels, order=2),
                ),
                stride,
                dropout,
            )
            for in_channels, out_channels, stride in _channel_pairs()
        ])
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _prepare_input(x, self.input_norm)
        supports = (
            self.adaptive_graph.physical_adjacency,
            self.adaptive_graph.adaptive_adjacency(),
        )
        for block in self.blocks:
            x = block(x, supports)
        return self.classifier(x.mean(dim=(2, 3)))

    def adjacency_components(self) -> dict[str, torch.Tensor]:
        return self.adaptive_graph.components()


class AGCRNFactorizedSTGCN(nn.Module):
    """Experiment 4: spectral experiment 3 plus node-factorised W."""

    def __init__(self, num_classes: int, dropout: float = 0.15, adaptive_dim: int = 10, pe_dim: int = 8):
        super().__init__()
        eigenvalues, eigenvectors = laplacian_eigenpairs(pe_dim)
        eigenvalues = torch.from_numpy(eigenvalues)
        encoding = torch.from_numpy(eigenvectors)
        self.input_norm = nn.BatchNorm1d(3 * NUM_NODES)
        self.adaptive_graph = GraphWaveNetAdaptiveAdjacency(adaptive_dim)
        self.blocks = nn.ModuleList([
            SpatialTemporalBlock(
                in_channels,
                out_channels,
                EigenvalueWeightedSpatial(
                    in_channels,
                    eigenvalues,
                    encoding,
                    NodeAdaptiveGraphWaveNetSupportProjection(in_channels, adaptive_dim, order=2),
                ),
                stride,
                dropout,
            )
            for in_channels, out_channels, stride in _channel_pairs()
        ])
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _prepare_input(x, self.input_norm)
        supports = (
            self.adaptive_graph.physical_adjacency,
            self.adaptive_graph.adaptive_adjacency(),
        )
        for block in self.blocks:
            x = block(x, supports, self.adaptive_graph.nodevec1)
        return self.classifier(x.mean(dim=(2, 3)))

    def adjacency_components(self) -> dict[str, torch.Tensor]:
        return self.adaptive_graph.components()


class _AGCRNBase(nn.Module):
    """AGCRN spatial operator with optional velocity channel and temporal gate.

    Reuses the AGCRN node-factorised projection as its spatial operator.
    `use_velocity` prepends frame-difference channels; `use_gate` inserts a
    TemporalGate after every block. Both default to off so the base behaviour is
    identical to :class:`AGCRNFactorizedSTGCN`.
    """

    def __init__(self, num_classes: int, dropout: float = 0.15, adaptive_dim: int = 10, pe_dim: int = 8, use_velocity: bool = False, use_gate: bool = False):
        super().__init__()
        eigenvalues, eigenvectors = laplacian_eigenpairs(pe_dim)
        eigenvalues = torch.from_numpy(eigenvalues)
        encoding = torch.from_numpy(eigenvectors)
        self.use_velocity = use_velocity
        self.use_gate = use_gate
        self.input_norm = nn.BatchNorm1d((6 if use_velocity else 3) * NUM_NODES)
        self.adaptive_graph = GraphWaveNetAdaptiveAdjacency(adaptive_dim)
        channel_pairs = [(6, 32, 1), *_channel_pairs()[1:]] if use_velocity else _channel_pairs()
        self.blocks = nn.ModuleList([
            SpatialTemporalBlock(
                in_channels,
                out_channels,
                EigenvalueWeightedSpatial(
                    in_channels,
                    eigenvalues,
                    encoding,
                    NodeAdaptiveGraphWaveNetSupportProjection(in_channels, adaptive_dim, order=2),
                ),
                stride,
                dropout,
            )
            for in_channels, out_channels, stride in channel_pairs
        ])
        if use_gate:
            self.gates = nn.ModuleList([TemporalGate(channels) for channels in (32, 64, 96, 128)])
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_velocity:
            x = _velocity_channels(x)
        x = _prepare_input(x, self.input_norm)
        supports = (
            self.adaptive_graph.physical_adjacency,
            self.adaptive_graph.adaptive_adjacency(),
        )
        for index, block in enumerate(self.blocks):
            x = block(x, supports, self.adaptive_graph.nodevec1)
            if self.use_gate:
                x = self.gates[index](x)
        return self.classifier(x.mean(dim=(2, 3)))

    def adjacency_components(self) -> dict[str, torch.Tensor]:
        return self.adaptive_graph.components()


class VelocityAGCRN(_AGCRNBase):
    """AGCRN + velocity feature channels."""

    def __init__(self, num_classes: int, dropout: float = 0.15, adaptive_dim: int = 10, pe_dim: int = 8):
        super().__init__(num_classes, dropout, adaptive_dim, pe_dim, use_velocity=True, use_gate=False)


class GatedAGCRN(_AGCRNBase):
    """AGCRN + temporal gate after every block."""

    def __init__(self, num_classes: int, dropout: float = 0.15, adaptive_dim: int = 10, pe_dim: int = 8):
        super().__init__(num_classes, dropout, adaptive_dim, pe_dim, use_velocity=False, use_gate=True)


class VelocityGatedAGCRN(_AGCRNBase):
    """AGCRN + velocity channels + temporal gate."""

    def __init__(self, num_classes: int, dropout: float = 0.15, adaptive_dim: int = 10, pe_dim: int = 8):
        super().__init__(num_classes, dropout, adaptive_dim, pe_dim, use_velocity=True, use_gate=True)


def build_experimental_model(
    name: str,
    num_classes: int,
    dropout: float,
    model_config: dict | None = None,
) -> nn.Module:
    config = model_config or {}
    pe_dim = int(config.get("pe_dim", 8))
    adaptive_dim = int(config.get("adaptive_dim", 10))
    attention_heads = int(config.get("attention_heads", config.get("gat_heads", 4)))
    if name == "spectral_pe_stgcn":
        return SpectralPESTGCN(num_classes, dropout, pe_dim)
    if name == "spectral_pe_qkv":
        return SpectralPEQKV(num_classes, dropout, pe_dim, attention_heads)
    if name == "spectral_pe_qkv_stable":
        return SpectralPEQKVStable(num_classes, dropout, pe_dim, attention_heads)
    if name == "gwnet_adaptive_support":
        return GraphWaveNetAdaptiveSTGCN(num_classes, dropout, adaptive_dim, pe_dim)
    if name == "agcrn_factorized_adjacency":
        return AGCRNFactorizedSTGCN(num_classes, dropout, adaptive_dim, pe_dim)
    if name == "velocity_agcrn":
        return VelocityAGCRN(num_classes, dropout, adaptive_dim, pe_dim)
    if name == "gated_agcrn":
        return GatedAGCRN(num_classes, dropout, adaptive_dim, pe_dim)
    if name == "velocity_gated_agcrn":
        return VelocityGatedAGCRN(num_classes, dropout, adaptive_dim, pe_dim)
    raise ValueError(f"unknown experimental model: {name}")
