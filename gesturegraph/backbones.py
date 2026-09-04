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


LEGACY_EXPERIMENTAL_MODEL_NAMES = (
    "spectral_pe_stgcn",
    "spectral_pe_qkv",
    "gwnet_adaptive_support",
    "agcrn_factorized_adjacency",
)

SE_IMPROVEMENT_MODEL_NAMES = (
    "stem_stgcn",
    "stem_linear_se",
    "stem_residual_mlp_se",
    "stem_residual_mlp_gated_se",
    "stem_residual_mlp_learnable_values_se",
)

SE_TRANSFER_MODEL_NAMES = (
    "stem_qkv_control",
    "stem_semantic_qkv",
    "stem_gwnet_control",
    "stem_semantic_gwnet",
    "stem_agcrn_control",
    "stem_semantic_agcrn",
)

SE_TRANSFER_V2_MODEL_NAMES = (
    "stem_stable_qkv_control",
    "stem_semantic_stable_qkv",
    "stem_gwnet_gated_control",
    "stem_semantic_gwnet_gated",
    "stem_agcrn_dynamic_control",
    "stem_semantic_agcrn_dynamic",
)

PURE_QKV_MODEL_NAMES = (
    "stem_pure_qkv_control",
    "stem_semantic_pure_qkv",
)

EXPERIMENTAL_MODEL_NAMES = (
    LEGACY_EXPERIMENTAL_MODEL_NAMES
    + SE_IMPROVEMENT_MODEL_NAMES
    + SE_TRANSFER_MODEL_NAMES
    + SE_TRANSFER_V2_MODEL_NAMES
    + PURE_QKV_MODEL_NAMES
)


def _prepare_input(x: torch.Tensor, input_norm: nn.BatchNorm1d) -> torch.Tensor:
    if x.ndim != 4 or x.shape[1] != 3 or x.shape[3] != NUM_NODES:
        raise ValueError(f"input must have shape [N, 3, T, {NUM_NODES}]")
    n, channels, frames, nodes = x.shape
    x = x.permute(0, 1, 3, 2).reshape(n, channels * nodes, frames)
    x = input_norm(x)
    return x.reshape(n, channels, nodes, frames).permute(0, 1, 3, 2)


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


class GatedDualSupportProjection(nn.Module):
    """Separate physical/dynamic diffusion branches with complementary fusion.

    Both branches use the same supports and order as the Graph WaveNet control.
    The physical branch always has a node-shared projection.  In the AGCRN
    variant only the dynamic branch obtains node-specific weights from the same
    low-rank factors that parameterise the adaptive support.  Node factors are
    never added to X and never alter the physical-graph projection.
    """

    def __init__(
        self,
        channels: int,
        embedding_dim: int,
        order: int = 2,
        factorize_dynamic: bool = False,
        dynamic_gate_init: float = 0.1,
    ):
        super().__init__()
        if order < 1:
            raise ValueError("diffusion order must be positive")
        if not 0.0 < dynamic_gate_init < 1.0:
            raise ValueError("dynamic_gate_init must be between zero and one")
        self.order = order
        self.factorize_dynamic = factorize_dynamic
        diffusion_channels = channels * (1 + order)
        self.physical_projection = nn.Conv2d(diffusion_channels, channels, 1, bias=False)
        if factorize_dynamic:
            self.dynamic_weights_pool = nn.Parameter(
                torch.empty(embedding_dim, diffusion_channels, channels)
            )
            self.dynamic_bias_pool = nn.Parameter(torch.zeros(embedding_dim, channels))
            nn.init.xavier_uniform_(self.dynamic_weights_pool)
            self.dynamic_projection = None
        else:
            self.dynamic_projection = nn.Conv2d(diffusion_channels, channels, 1, bias=False)
            self.register_parameter("dynamic_weights_pool", None)
            self.register_parameter("dynamic_bias_pool", None)
        initial_logit = math.log(dynamic_gate_init / (1.0 - dynamic_gate_init))
        self.dynamic_gate_logit = nn.Parameter(torch.tensor(initial_logit))

    def _diffuse(self, x: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
        outputs = [x]
        propagated = x
        for _ in range(self.order):
            propagated = torch.einsum("nctv,vw->nctw", propagated, support)
            outputs.append(propagated)
        return torch.cat(outputs, dim=1)

    def dynamic_gate(self) -> torch.Tensor:
        return self.dynamic_gate_logit.sigmoid()

    def forward(
        self,
        x: torch.Tensor,
        adjacency: tuple[torch.Tensor, torch.Tensor] | None = None,
        node_embeddings: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if adjacency is None or len(adjacency) != 2:
            raise ValueError("physical and adaptive supports are required")
        physical_features = self._diffuse(x, adjacency[0])
        dynamic_features = self._diffuse(x, adjacency[1])
        physical = self.physical_projection(physical_features)
        if self.factorize_dynamic:
            if node_embeddings is None:
                raise ValueError("node factors are required for dynamic weight factorisation")
            weights = torch.einsum("vd,dio->vio", node_embeddings, self.dynamic_weights_pool)
            bias = torch.einsum("vd,do->vo", node_embeddings, self.dynamic_bias_pool)
            dynamic = torch.einsum("nitv,vio->notv", dynamic_features, weights)
            dynamic = dynamic + bias.transpose(0, 1).unsqueeze(0).unsqueeze(2)
        else:
            dynamic = self.dynamic_projection(dynamic_features)
        gate = self.dynamic_gate()
        return (1.0 - gate) * physical + gate * dynamic


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


class SemanticSpectralSpatial(nn.Module):
    """Inject topology-preserving linear PE plus an optional semantic MLP.

    The fixed Laplacian eigenvectors remain the node identity source.  A
    per-layer positive residual gate can reweight spectral frequencies without
    replacing the physical eigenpairs.  The linear branch preserves a direct
    path from spectral coordinates, while the zero-safe residual MLP learns
    task-specific node semantics.
    """

    def __init__(
        self,
        channels: int,
        eigenvalues: torch.Tensor,
        positional_encoding: torch.Tensor,
        spatial: nn.Module,
        semantic_hidden: int | None = None,
        learnable_spectral_gate: bool = False,
        direct_learnable_spectral_values: bool = False,
    ):
        super().__init__()
        if learnable_spectral_gate and direct_learnable_spectral_values:
            raise ValueError(
                "bounded spectral gates and direct learnable spectral values are mutually exclusive"
            )
        self.register_buffer("spectral_eigenvalues", eigenvalues.clone())
        self.register_buffer("spectral_encoding", positional_encoding.clone())
        dimensions = positional_encoding.shape[1]
        self.position_projection = nn.Linear(dimensions, channels, bias=False)
        self.semantic_projection = None
        if semantic_hidden is not None:
            if semantic_hidden < 1:
                raise ValueError("semantic_hidden must be positive")
            self.semantic_projection = nn.Sequential(
                nn.Linear(dimensions, semantic_hidden),
                nn.ReLU(inplace=True),
                nn.Linear(semantic_hidden, channels, bias=False),
            )
            self.semantic_scale = nn.Parameter(torch.tensor(0.1))
        else:
            self.register_parameter("semantic_scale", None)
        if learnable_spectral_gate:
            self.spectral_gate_delta = nn.Parameter(torch.zeros(dimensions))
        else:
            self.register_parameter("spectral_gate_delta", None)
        if direct_learnable_spectral_values:
            # The Laplacian eigenvectors stay fixed. Only their K diagonal
            # spectral weights are optimized, starting from the graph's
            # original eigenvalues; no K x K mixing matrix is introduced.
            self.learnable_spectral_values = nn.Parameter(eigenvalues.clone())
        else:
            self.register_parameter("learnable_spectral_values", None)
        self.position_scale = nn.Parameter(torch.tensor(0.1))
        self.spatial = spatial

    def spectral_weights(self) -> torch.Tensor:
        if self.learnable_spectral_values is not None:
            return self.learnable_spectral_values
        if self.spectral_gate_delta is None:
            return self.spectral_eigenvalues
        # Bounded positive multiplier: exp(2*tanh(delta)) is in [e^-2, e^2]
        # and equals one at initialisation, so the model starts from the fixed
        # Laplacian eigenvalue weighting used by the established baseline.
        multiplier = torch.exp(2.0 * torch.tanh(self.spectral_gate_delta))
        return self.spectral_eigenvalues * multiplier

    def positional_features(self) -> torch.Tensor:
        weighted = self.spectral_encoding * self.spectral_weights().unsqueeze(0)
        position = self.position_projection(weighted)
        if self.semantic_projection is not None:
            position = position + self.semantic_scale * self.semantic_projection(weighted)
        return position

    def forward(
        self,
        x: torch.Tensor,
        adjacency: torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None = None,
        node_embeddings: torch.Tensor | None = None,
    ) -> torch.Tensor:
        channels = x.shape[1]
        nodes = x.shape[3]
        position = self.positional_features().transpose(0, 1).view(1, channels, 1, nodes)
        return self.spatial(x + self.position_scale * position, adjacency, node_embeddings)


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


class PureMultiHeadQKVSpatial(nn.Module):
    """Full-joint QKV with no adjacency bias or mask.

    Spatial identity, when requested by the backbone, enters only through the
    semantic spectral representation before Q/K/V.  This keeps the attention
    logits entirely content-driven while preserving the established four-head
    projections and 0.1-initialised residual path.
    """

    def __init__(self, channels: int, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        if heads < 1:
            raise ValueError("heads must be positive")
        self.heads = heads
        self.head_dim = max(8, math.ceil(channels / heads))
        projection_width = heads * self.head_dim
        self.query_projection = nn.Linear(channels, projection_width, bias=False)
        self.key_projection = nn.Linear(channels, projection_width, bias=False)
        self.value_projection = nn.Linear(channels, projection_width, bias=False)
        self.output_projection = nn.Linear(projection_width, channels, bias=False)
        self.attention_dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(dropout)
        self.residual_scale = nn.Parameter(torch.tensor(0.1))

    def _project(self, layer: nn.Linear, features: torch.Tensor) -> torch.Tensor:
        batch, frames, nodes, _ = features.shape
        projected = layer(features).view(batch, frames, nodes, self.heads, self.head_dim)
        return projected.permute(0, 1, 3, 2, 4)

    def attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        features = x.permute(0, 2, 3, 1)
        query = self._project(self.query_projection, features)
        key = self._project(self.key_projection, features)
        scores = torch.einsum("bthid,bthjd->bthij", query, key) / math.sqrt(self.head_dim)
        return scores.softmax(dim=-1)

    def forward(
        self,
        x: torch.Tensor,
        adjacency: torch.Tensor | None = None,
        node_embeddings: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del adjacency, node_embeddings
        features = x.permute(0, 2, 3, 1)
        value = self._project(self.value_projection, features)
        attention = self.attention_dropout(self.attention_weights(x))
        attended = torch.einsum("bthij,bthjd->bthid", attention, value)
        batch, frames, _, nodes, _ = attended.shape
        attended = attended.permute(0, 1, 3, 2, 4).reshape(batch, frames, nodes, -1)
        output = self.output_projection(attended).permute(0, 3, 1, 2).contiguous()
        return x + self.residual_scale * self.output_dropout(output)


class StableMultiHeadQKVSpatial(nn.Module):
    """QKV attention with pre-norm, positive per-head temperature and unit residual gain."""

    def __init__(self, channels: int, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        if heads < 1:
            raise ValueError("heads must be positive")
        self.heads = heads
        self.head_dim = max(8, math.ceil(channels / heads))
        self.register_buffer("physical_adjacency", torch.from_numpy(binary_adjacency(include_self=True)))
        projection_width = heads * self.head_dim
        self.pre_norm = nn.LayerNorm(channels)
        self.query_projection = nn.Linear(channels, projection_width, bias=False)
        self.key_projection = nn.Linear(channels, projection_width, bias=False)
        self.value_projection = nn.Linear(channels, projection_width, bias=False)
        self.output_projection = nn.Linear(projection_width, channels, bias=False)
        self.attention_dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(dropout)
        self.physical_bias = nn.Parameter(torch.full((heads,), 0.5))
        self.log_temperature = nn.Parameter(torch.zeros(heads))
        self.residual_scale = nn.Parameter(torch.tensor(1.0))

    def temperatures(self) -> torch.Tensor:
        return self.log_temperature.exp()

    def forward(
        self,
        x: torch.Tensor,
        adjacency: torch.Tensor | None = None,
        node_embeddings: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del adjacency, node_embeddings
        batch, _, frames, nodes = x.shape
        features = self.pre_norm(x.permute(0, 2, 3, 1))

        def project(layer: nn.Linear) -> torch.Tensor:
            projected = layer(features).view(batch, frames, nodes, self.heads, self.head_dim)
            return projected.permute(0, 1, 3, 2, 4)

        query = project(self.query_projection)
        key = project(self.key_projection)
        value = project(self.value_projection)
        scores = torch.einsum("bthid,bthjd->bthij", query, key) / math.sqrt(self.head_dim)
        scores = scores / self.temperatures().view(1, 1, self.heads, 1, 1)
        scores = scores + self.physical_bias.view(1, 1, self.heads, 1, 1) * self.physical_adjacency.view(
            1, 1, 1, nodes, nodes
        )
        attention = self.attention_dropout(scores.softmax(dim=-1))
        attended = torch.einsum("bthij,bthjd->bthid", attention, value)
        attended = attended.permute(0, 1, 3, 2, 4).reshape(batch, frames, nodes, -1)
        output = self.output_projection(attended).permute(0, 3, 1, 2).contiguous()
        return x + self.residual_scale * self.output_dropout(output)


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


def _stem_channel_pairs(stem_channels: int = 32) -> list[tuple[int, int, int]]:
    return [(stem_channels, stem_channels, 1), (stem_channels, 64, 2), (64, 96, 2), (96, 128, 1)]


class InputFeatureStem(nn.Module):
    """Lift per-joint XYZ features before any spatial position is injected."""

    def __init__(self, channels: int = 32):
        super().__init__()
        if channels < 1:
            raise ValueError("stem channels must be positive")
        self.network = nn.Sequential(
            nn.Conv2d(3, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class StemSTGCN(nn.Module):
    """Matched control: 3->C stem followed by fixed-adjacency ST-GCN."""

    def __init__(self, num_classes: int, dropout: float = 0.15, stem_channels: int = 32):
        super().__init__()
        self.register_buffer("adjacency", torch.from_numpy(normalized_adjacency()))
        self.input_norm = nn.BatchNorm1d(3 * NUM_NODES)
        self.input_stem = InputFeatureStem(stem_channels)
        self.blocks = nn.ModuleList([
            SpatialTemporalBlock(
                in_channels,
                out_channels,
                AdjacencyAggregate(),
                stride,
                dropout,
            )
            for in_channels, out_channels, stride in _stem_channel_pairs(stem_channels)
        ])
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_stem(_prepare_input(x, self.input_norm))
        for block in self.blocks:
            x = block(x, self.adjacency)
        return self.classifier(x.mean(dim=(2, 3)))


class StemSemanticSESTGCN(nn.Module):
    """Stem-first ST-GCN with linear/residual-MLP spectral node semantics."""

    def __init__(
        self,
        num_classes: int,
        dropout: float = 0.15,
        pe_dim: int = 21,
        stem_channels: int = 32,
        semantic_hidden: int | None = None,
        learnable_spectral_gate: bool = False,
        direct_learnable_spectral_values: bool = False,
    ):
        super().__init__()
        eigenvalues, eigenvectors = laplacian_eigenpairs(pe_dim)
        eigenvalues = torch.from_numpy(eigenvalues)
        encoding = torch.from_numpy(eigenvectors)
        self.register_buffer("adjacency", torch.from_numpy(normalized_adjacency()))
        self.input_norm = nn.BatchNorm1d(3 * NUM_NODES)
        self.input_stem = InputFeatureStem(stem_channels)
        self.blocks = nn.ModuleList([
            SpatialTemporalBlock(
                in_channels,
                out_channels,
                SemanticSpectralSpatial(
                    in_channels,
                    eigenvalues,
                    encoding,
                    AdjacencyAggregate(),
                    semantic_hidden=semantic_hidden,
                    learnable_spectral_gate=learnable_spectral_gate,
                    direct_learnable_spectral_values=direct_learnable_spectral_values,
                ),
                stride,
                dropout,
            )
            for in_channels, out_channels, stride in _stem_channel_pairs(stem_channels)
        ])
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_stem(_prepare_input(x, self.input_norm))
        for block in self.blocks:
            x = block(x, self.adjacency)
        return self.classifier(x.mean(dim=(2, 3)))


def _semantic_wrapper(
    spatial: nn.Module,
    channels: int,
    eigenvalues: torch.Tensor | None,
    encoding: torch.Tensor | None,
    semantic_hidden: int,
) -> nn.Module:
    if eigenvalues is None or encoding is None:
        return spatial
    return SemanticSpectralSpatial(
        channels,
        eigenvalues,
        encoding,
        spatial,
        semantic_hidden=semantic_hidden,
    )


class StemQKVBackbone(nn.Module):
    """Stem-first QKV with a matched no-SE or semantic-SE spatial input."""

    def __init__(
        self,
        num_classes: int,
        dropout: float = 0.15,
        pe_dim: int = 21,
        stem_channels: int = 32,
        semantic_hidden: int = 64,
        heads: int = 4,
        use_semantic_se: bool = False,
        pure_attention: bool = False,
    ):
        super().__init__()
        eigenvalues = encoding = None
        if use_semantic_se:
            values, vectors = laplacian_eigenpairs(pe_dim)
            eigenvalues, encoding = torch.from_numpy(values), torch.from_numpy(vectors)
        self.input_norm = nn.BatchNorm1d(3 * NUM_NODES)
        self.input_stem = InputFeatureStem(stem_channels)
        attention_type = PureMultiHeadQKVSpatial if pure_attention else MultiHeadQKVSpatial
        self.blocks = nn.ModuleList([
            SpatialTemporalBlock(
                in_channels,
                out_channels,
                _semantic_wrapper(
                    attention_type(in_channels, heads=heads, dropout=dropout),
                    in_channels,
                    eigenvalues,
                    encoding,
                    semantic_hidden,
                ),
                stride,
                dropout,
            )
            for in_channels, out_channels, stride in _stem_channel_pairs(stem_channels)
        ])
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_stem(_prepare_input(x, self.input_norm))
        for block in self.blocks:
            x = block(x)
        return self.classifier(x.mean(dim=(2, 3)))


class StemAdaptiveBackbone(nn.Module):
    """Matched stem-first Graph WaveNet/AGCRN with optional semantic SE."""

    def __init__(
        self,
        num_classes: int,
        dropout: float = 0.15,
        pe_dim: int = 21,
        stem_channels: int = 32,
        semantic_hidden: int = 64,
        adaptive_dim: int = 10,
        node_factorized: bool = False,
        use_semantic_se: bool = False,
    ):
        super().__init__()
        eigenvalues = encoding = None
        if use_semantic_se:
            values, vectors = laplacian_eigenpairs(pe_dim)
            eigenvalues, encoding = torch.from_numpy(values), torch.from_numpy(vectors)
        self.node_factorized = node_factorized
        self.input_norm = nn.BatchNorm1d(3 * NUM_NODES)
        self.input_stem = InputFeatureStem(stem_channels)
        self.adaptive_graph = GraphWaveNetAdaptiveAdjacency(adaptive_dim)
        blocks = []
        for in_channels, out_channels, stride in _stem_channel_pairs(stem_channels):
            operator = (
                NodeAdaptiveGraphWaveNetSupportProjection(in_channels, adaptive_dim, order=2)
                if node_factorized
                else GraphWaveNetSupportProjection(in_channels, order=2)
            )
            blocks.append(SpatialTemporalBlock(
                in_channels,
                out_channels,
                _semantic_wrapper(
                    operator,
                    in_channels,
                    eigenvalues,
                    encoding,
                    semantic_hidden,
                ),
                stride,
                dropout,
            ))
        self.blocks = nn.ModuleList(blocks)
        self.classifier = nn.Linear(128, num_classes)

    def forward_feature_map(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_stem(_prepare_input(x, self.input_norm))
        supports = (
            self.adaptive_graph.physical_adjacency,
            self.adaptive_graph.adaptive_adjacency(),
        )
        node_embeddings = self.adaptive_graph.nodevec1 if self.node_factorized else None
        for block in self.blocks:
            x = block(x, supports, node_embeddings)
        return x

    def encode(
        self,
        x: torch.Tensor,
        valid_lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return the pooled 128-D representation, optionally excluding padding.

        The stem AGCRN has two stride-two temporal blocks.  A causal prefix with
        ``valid_lengths`` input frames therefore contributes ceil(length / 4)
        output positions.  Padding is never included in the temporal mean.
        """
        features = self.forward_feature_map(x)
        if valid_lengths is None:
            return features.mean(dim=(2, 3))
        output_lengths = torch.div(
            valid_lengths.to(device=features.device, dtype=torch.long) + 3,
            4,
            rounding_mode="floor",
        ).clamp(min=1, max=features.shape[2])
        mask = torch.arange(features.shape[2], device=features.device).unsqueeze(0)
        mask = (mask < output_lengths.unsqueeze(1)).to(features.dtype)
        pooled = (features * mask[:, None, :, None]).sum(dim=(2, 3))
        return pooled / (output_lengths.to(features.dtype) * features.shape[3]).unsqueeze(1)

    def forward(
        self,
        x: torch.Tensor,
        valid_lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.classifier(self.encode(x, valid_lengths))

    def adjacency_components(self) -> dict[str, torch.Tensor]:
        return self.adaptive_graph.components()


class StemStableQKVBackbone(nn.Module):
    """Stem-first stable QKV with an optional matched semantic-SE wrapper."""

    def __init__(
        self,
        num_classes: int,
        dropout: float = 0.15,
        pe_dim: int = 21,
        stem_channels: int = 32,
        semantic_hidden: int = 64,
        heads: int = 4,
        use_semantic_se: bool = False,
    ):
        super().__init__()
        eigenvalues = encoding = None
        if use_semantic_se:
            values, vectors = laplacian_eigenpairs(pe_dim)
            eigenvalues, encoding = torch.from_numpy(values), torch.from_numpy(vectors)
        self.input_norm = nn.BatchNorm1d(3 * NUM_NODES)
        self.input_stem = InputFeatureStem(stem_channels)
        self.blocks = nn.ModuleList([
            SpatialTemporalBlock(
                in_channels,
                out_channels,
                _semantic_wrapper(
                    StableMultiHeadQKVSpatial(in_channels, heads=heads, dropout=dropout),
                    in_channels,
                    eigenvalues,
                    encoding,
                    semantic_hidden,
                ),
                stride,
                dropout,
            )
            for in_channels, out_channels, stride in _stem_channel_pairs(stem_channels)
        ])
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_stem(_prepare_input(x, self.input_norm))
        for block in self.blocks:
            x = block(x)
        return self.classifier(x.mean(dim=(2, 3)))


class StemGatedAdaptiveBackbone(nn.Module):
    """Continuous GWN/AGCRN ladder with gated physical and dynamic branches."""

    def __init__(
        self,
        num_classes: int,
        dropout: float = 0.15,
        pe_dim: int = 21,
        stem_channels: int = 32,
        semantic_hidden: int = 64,
        adaptive_dim: int = 10,
        factorize_dynamic: bool = False,
        use_semantic_se: bool = False,
    ):
        super().__init__()
        eigenvalues = encoding = None
        if use_semantic_se:
            values, vectors = laplacian_eigenpairs(pe_dim)
            eigenvalues, encoding = torch.from_numpy(values), torch.from_numpy(vectors)
        self.factorize_dynamic = factorize_dynamic
        self.input_norm = nn.BatchNorm1d(3 * NUM_NODES)
        self.input_stem = InputFeatureStem(stem_channels)
        self.adaptive_graph = GraphWaveNetAdaptiveAdjacency(adaptive_dim)
        self.blocks = nn.ModuleList([
            SpatialTemporalBlock(
                in_channels,
                out_channels,
                _semantic_wrapper(
                    GatedDualSupportProjection(
                        in_channels,
                        adaptive_dim,
                        order=2,
                        factorize_dynamic=factorize_dynamic,
                        dynamic_gate_init=0.1,
                    ),
                    in_channels,
                    eigenvalues,
                    encoding,
                    semantic_hidden,
                ),
                stride,
                dropout,
            )
            for in_channels, out_channels, stride in _stem_channel_pairs(stem_channels)
        ])
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_stem(_prepare_input(x, self.input_norm))
        supports = (
            self.adaptive_graph.physical_adjacency,
            self.adaptive_graph.adaptive_adjacency(),
        )
        node_factors = self.adaptive_graph.nodevec1 if self.factorize_dynamic else None
        for block in self.blocks:
            x = block(x, supports, node_factors)
        return self.classifier(x.mean(dim=(2, 3)))

    def adjacency_components(self) -> dict[str, torch.Tensor]:
        components = self.adaptive_graph.components()
        gates = []
        for block in self.blocks:
            spatial = block.spatial
            operator = spatial.spatial if isinstance(spatial, SemanticSpectralSpatial) else spatial
            gates.append(operator.dynamic_gate().detach().cpu())
        components["dynamic_branch_gates"] = torch.stack(gates)
        return components


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
    stem_channels = int(config.get("stem_channels", 32))
    semantic_hidden = int(config.get("semantic_hidden", 64))
    if name == "spectral_pe_stgcn":
        return SpectralPESTGCN(num_classes, dropout, pe_dim)
    if name == "spectral_pe_qkv":
        return SpectralPEQKV(num_classes, dropout, pe_dim, attention_heads)
    if name == "gwnet_adaptive_support":
        return GraphWaveNetAdaptiveSTGCN(num_classes, dropout, adaptive_dim, pe_dim)
    if name == "agcrn_factorized_adjacency":
        return AGCRNFactorizedSTGCN(num_classes, dropout, adaptive_dim, pe_dim)
    if name == "stem_stgcn":
        return StemSTGCN(num_classes, dropout, stem_channels)
    if name == "stem_linear_se":
        return StemSemanticSESTGCN(num_classes, dropout, pe_dim, stem_channels)
    if name == "stem_residual_mlp_se":
        return StemSemanticSESTGCN(
            num_classes,
            dropout,
            pe_dim,
            stem_channels,
            semantic_hidden=semantic_hidden,
        )
    if name == "stem_residual_mlp_gated_se":
        return StemSemanticSESTGCN(
            num_classes,
            dropout,
            pe_dim,
            stem_channels,
            semantic_hidden=semantic_hidden,
            learnable_spectral_gate=True,
        )
    if name == "stem_residual_mlp_learnable_values_se":
        return StemSemanticSESTGCN(
            num_classes,
            dropout,
            pe_dim,
            stem_channels,
            semantic_hidden=semantic_hidden,
            direct_learnable_spectral_values=True,
        )
    if name in {"stem_qkv_control", "stem_semantic_qkv"}:
        return StemQKVBackbone(
            num_classes,
            dropout,
            pe_dim,
            stem_channels,
            semantic_hidden,
            attention_heads,
            use_semantic_se=name == "stem_semantic_qkv",
        )
    if name in {"stem_pure_qkv_control", "stem_semantic_pure_qkv"}:
        return StemQKVBackbone(
            num_classes,
            dropout,
            pe_dim,
            stem_channels,
            semantic_hidden,
            attention_heads,
            use_semantic_se=name == "stem_semantic_pure_qkv",
            pure_attention=True,
        )
    if name in {"stem_gwnet_control", "stem_semantic_gwnet"}:
        return StemAdaptiveBackbone(
            num_classes,
            dropout,
            pe_dim,
            stem_channels,
            semantic_hidden,
            adaptive_dim,
            node_factorized=False,
            use_semantic_se=name == "stem_semantic_gwnet",
        )
    if name in {"stem_agcrn_control", "stem_semantic_agcrn"}:
        return StemAdaptiveBackbone(
            num_classes,
            dropout,
            pe_dim,
            stem_channels,
            semantic_hidden,
            adaptive_dim,
            node_factorized=True,
            use_semantic_se=name == "stem_semantic_agcrn",
        )
    if name in {"stem_stable_qkv_control", "stem_semantic_stable_qkv"}:
        return StemStableQKVBackbone(
            num_classes,
            dropout,
            pe_dim,
            stem_channels,
            semantic_hidden,
            attention_heads,
            use_semantic_se=name == "stem_semantic_stable_qkv",
        )
    if name in {"stem_gwnet_gated_control", "stem_semantic_gwnet_gated"}:
        return StemGatedAdaptiveBackbone(
            num_classes,
            dropout,
            pe_dim,
            stem_channels,
            semantic_hidden,
            adaptive_dim,
            factorize_dynamic=False,
            use_semantic_se=name == "stem_semantic_gwnet_gated",
        )
    if name in {"stem_agcrn_dynamic_control", "stem_semantic_agcrn_dynamic"}:
        return StemGatedAdaptiveBackbone(
            num_classes,
            dropout,
            pe_dim,
            stem_channels,
            semantic_hidden,
            adaptive_dim,
            factorize_dynamic=True,
            use_semantic_se=name == "stem_semantic_agcrn_dynamic",
        )
    raise ValueError(f"unknown experimental model: {name}")
