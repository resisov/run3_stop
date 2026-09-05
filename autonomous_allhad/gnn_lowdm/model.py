from __future__ import annotations

import torch
from torch import nn


class EdgeMessageBlock(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.message = nn.Sequential(
            nn.Linear(2 * hidden + 4, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
        )
        self.update = nn.Sequential(
            nn.Linear(3 * hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.norm = nn.LayerNorm(hidden)

    def forward(
        self,
        hidden: torch.Tensor,
        eta: torch.Tensor,
        phi: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        batch, nodes, width = hidden.shape
        hi = hidden[:, :, None, :].expand(batch, nodes, nodes, width)
        hj = hidden[:, None, :, :].expand(batch, nodes, nodes, width)
        deta = eta[:, :, None] - eta[:, None, :]
        dphi = phi[:, :, None] - phi[:, None, :]
        edge = torch.stack(
            (deta, torch.sin(dphi), torch.cos(dphi), torch.sqrt(deta.square() + torch.sin(dphi).square() + 1e-8)),
            dim=-1,
        )
        messages = self.message(torch.cat((hi, hj, edge), dim=-1))
        pair_mask = mask[:, :, None] & mask[:, None, :]
        eye = torch.eye(nodes, dtype=torch.bool, device=hidden.device)[None, :, :]
        pair_mask &= ~eye
        messages = messages * pair_mask[..., None]
        count = pair_mask.sum(dim=2, keepdim=True).clamp(min=1)
        mean = messages.sum(dim=2) / count
        masked_for_max = messages.masked_fill(~pair_mask[..., None], -1.0e9)
        maximum = masked_for_max.max(dim=2).values
        maximum = torch.where(mask[:, :, None], maximum, torch.zeros_like(maximum))
        updated = self.update(torch.cat((hidden, mean, maximum), dim=-1))
        return self.norm(hidden + updated) * mask[..., None]


class JetGraphClassifier(nn.Module):
    """Permutation-invariant dense jet graph for small event graphs."""

    def __init__(
        self,
        node_features: int = 6,
        global_features: int = 9,
        hidden: int = 32,
        message_layers: int = 2,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(node_features, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
        )
        self.blocks = nn.ModuleList(EdgeMessageBlock(hidden) for _ in range(message_layers))
        self.head = nn.Sequential(
            nn.Linear(2 * hidden + global_features, 2 * hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(2 * hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def forward(
        self,
        nodes: torch.Tensor,
        mask: torch.Tensor,
        eta: torch.Tensor,
        phi: torch.Tensor,
        globals_: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self.encoder(nodes) * mask[..., None]
        for block in self.blocks:
            hidden = block(hidden, eta, phi, mask)
        count = mask.sum(dim=1, keepdim=True).clamp(min=1)
        mean = hidden.sum(dim=1) / count
        maximum = hidden.masked_fill(~mask[..., None], -1.0e9).max(dim=1).values
        maximum = torch.where(torch.isfinite(maximum), maximum, torch.zeros_like(maximum))
        return self.head(torch.cat((mean, maximum, globals_), dim=-1)).squeeze(-1)


def wrapped_edge_geometry(
    eta: torch.Tensor, phi: torch.Tensor
) -> torch.Tensor:
    """Return (deta, sin(dphi), cos(dphi), dR) with a true wrapped dphi.

    This is kept separate from :class:`EdgeMessageBlock` so checkpoints made
    with the original edge convention retain their exact inference semantics.
    """
    deta = eta[:, :, None] - eta[:, None, :]
    raw_dphi = phi[:, :, None] - phi[:, None, :]
    sin_dphi = torch.sin(raw_dphi)
    cos_dphi = torch.cos(raw_dphi)
    dphi = torch.atan2(sin_dphi, cos_dphi)
    delta_r = torch.sqrt(deta.square() + dphi.square() + 1.0e-8)
    return torch.stack((deta, sin_dphi, cos_dphi, delta_r), dim=-1)


class WrappedEdgeMessageBlock(nn.Module):
    """Message block using the physical wrapped angular distance."""

    def __init__(self, hidden: int, dropout: float) -> None:
        super().__init__()
        self.message = nn.Sequential(
            nn.Linear(2 * hidden + 4, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
        )
        self.update = nn.Sequential(
            nn.Linear(3 * hidden, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
        )
        self.norm = nn.LayerNorm(hidden)

    def forward(
        self,
        hidden: torch.Tensor,
        eta: torch.Tensor,
        phi: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        batch, nodes, width = hidden.shape
        hi = hidden[:, :, None, :].expand(batch, nodes, nodes, width)
        hj = hidden[:, None, :, :].expand(batch, nodes, nodes, width)
        messages = self.message(
            torch.cat((hi, hj, wrapped_edge_geometry(eta, phi)), dim=-1)
        )
        pair_mask = mask[:, :, None] & mask[:, None, :]
        eye = torch.eye(nodes, dtype=torch.bool, device=hidden.device)[None, :, :]
        pair_mask &= ~eye
        messages = messages * pair_mask[..., None]
        count = pair_mask.sum(dim=2, keepdim=True).clamp(min=1)
        mean = messages.sum(dim=2) / count
        masked_for_max = messages.masked_fill(~pair_mask[..., None], -1.0e9)
        maximum = masked_for_max.max(dim=2).values
        maximum = torch.where(mask[:, :, None], maximum, torch.zeros_like(maximum))
        updated = self.update(torch.cat((hidden, mean, maximum), dim=-1))
        return self.norm(hidden + updated) * mask[..., None]


class PhysicsInformedJetGraphClassifier(nn.Module):
    """Compact common-score GNN with global conditioning during message passing."""

    def __init__(
        self,
        node_features: int = 6,
        global_features: int = 20,
        hidden: int = 64,
        message_layers: int = 2,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(node_features, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
        )
        self.global_context = nn.Sequential(
            nn.Linear(global_features, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
        )
        self.blocks = nn.ModuleList(
            WrappedEdgeMessageBlock(hidden, dropout)
            for _ in range(message_layers)
        )
        self.head = nn.Sequential(
            nn.Linear(3 * hidden + global_features, 2 * hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(2 * hidden, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(
        self,
        nodes: torch.Tensor,
        mask: torch.Tensor,
        eta: torch.Tensor,
        phi: torch.Tensor,
        globals_: torch.Tensor,
    ) -> torch.Tensor:
        context = self.global_context(globals_)
        hidden = (self.encoder(nodes) + context[:, None, :]) * mask[..., None]
        for block in self.blocks:
            hidden = block(hidden, eta, phi, mask)
        count = mask.sum(dim=1, keepdim=True).clamp(min=1)
        mean = hidden.sum(dim=1) / count
        maximum = hidden.masked_fill(~mask[..., None], -1.0e9).max(dim=1).values
        maximum = torch.where(
            torch.isfinite(maximum), maximum, torch.zeros_like(maximum)
        )
        return self.head(
            torch.cat((mean, maximum, context, globals_), dim=-1)
        ).squeeze(-1)


class JetGraphProcessAwareClassifier(nn.Module):
    """Jet GNN with a binary signal head and an auxiliary process head.

    The auxiliary classes are expected to be signal, TT/ST, Z->nunu,
    W->lnu, and other background.  The binary score remains the search
    discriminant, while the process head forces the shared representation to
    retain information that separates the three dominant background families.
    """

    def __init__(
        self,
        node_features: int = 6,
        global_features: int = 9,
        hidden: int = 64,
        message_layers: int = 3,
        dropout: float = 0.15,
        process_classes: int = 5,
    ) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(node_features, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
        )
        self.blocks = nn.ModuleList(EdgeMessageBlock(hidden) for _ in range(message_layers))
        pooled_width = 2 * hidden + global_features
        self.shared_head = nn.Sequential(
            nn.Linear(pooled_width, 2 * hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(2 * hidden, hidden),
            nn.SiLU(),
        )
        self.binary_head = nn.Linear(hidden, 1)
        self.process_head = nn.Linear(hidden, process_classes)

    def encode_shared(
        self,
        nodes: torch.Tensor,
        mask: torch.Tensor,
        eta: torch.Tensor,
        phi: torch.Tensor,
        globals_: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self.encoder(nodes) * mask[..., None]
        for block in self.blocks:
            hidden = block(hidden, eta, phi, mask)
        count = mask.sum(dim=1, keepdim=True).clamp(min=1)
        mean = hidden.sum(dim=1) / count
        maximum = hidden.masked_fill(~mask[..., None], -1.0e9).max(dim=1).values
        maximum = torch.where(
            torch.isfinite(maximum), maximum, torch.zeros_like(maximum)
        )
        return self.shared_head(torch.cat((mean, maximum, globals_), dim=-1))

    def forward(
        self,
        nodes: torch.Tensor,
        mask: torch.Tensor,
        eta: torch.Tensor,
        phi: torch.Tensor,
        globals_: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        shared = self.encode_shared(nodes, mask, eta, phi, globals_)
        return self.binary_head(shared).squeeze(-1), self.process_head(shared)


class JetGraphProcessTopologyClassifier(JetGraphProcessAwareClassifier):
    """Process-aware classifier with an auxiliary signal-topology head.

    The topology target is background, T2tt, T2bW, or T2tb.  Its output is
    used only as a representation-learning loss; the search discriminant still
    comes from the binary and background-process heads.
    """

    def __init__(
        self,
        node_features: int = 6,
        global_features: int = 9,
        hidden: int = 64,
        message_layers: int = 3,
        dropout: float = 0.15,
        process_classes: int = 5,
        topology_classes: int = 4,
    ) -> None:
        super().__init__(
            node_features=node_features,
            global_features=global_features,
            hidden=hidden,
            message_layers=message_layers,
            dropout=dropout,
            process_classes=process_classes,
        )
        self.topology_head = nn.Linear(hidden, topology_classes)

    def forward(
        self,
        nodes: torch.Tensor,
        mask: torch.Tensor,
        eta: torch.Tensor,
        phi: torch.Tensor,
        globals_: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        shared = self.encode_shared(nodes, mask, eta, phi, globals_)
        return (
            self.binary_head(shared).squeeze(-1),
            self.process_head(shared),
            self.topology_head(shared),
        )


class MassConditionedJetGraphClassifier(nn.Module):
    """Process-aware jet GNN with a mass/topology-conditioned binary head.

    The event encoder is independent of the tested signal hypothesis.  A small
    hypernetwork maps ``(mStop, mLSP, deltaM, topology one-hot)`` to a linear
    discriminant in the event embedding.  Besides preventing background events
    from receiving a privileged null hypothesis, this factorization makes it
    possible to score many signal hypotheses without rerunning message passing.

    ``conditioned_logits`` accepts either one condition per event (shape
    ``[batch, condition_features]``), several adversarial conditions per event
    (``[batch, hypotheses, condition_features]``), or a common hypothesis grid
    (``[hypotheses, condition_features]`` through ``grid_logits``).
    """

    def __init__(
        self,
        node_features: int = 6,
        global_features: int = 30,
        condition_features: int = 6,
        hidden: int = 96,
        message_layers: int = 3,
        dropout: float = 0.12,
        process_classes: int = 5,
        event_layer_norm: bool = True,
    ) -> None:
        super().__init__()
        self.hidden = hidden
        self.encoder = nn.Sequential(
            nn.Linear(node_features, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
        )
        self.blocks = nn.ModuleList(EdgeMessageBlock(hidden) for _ in range(message_layers))
        event_layers: list[nn.Module] = [
            nn.Linear(2 * hidden + global_features, 2 * hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(2 * hidden, hidden),
            nn.SiLU(),
        ]
        if event_layer_norm:
            event_layers.append(nn.LayerNorm(hidden))
        self.event_head = nn.Sequential(*event_layers)
        self.event_query = nn.Linear(hidden, hidden)
        self.event_bias = nn.Linear(hidden, 1)
        self.condition_key = nn.Sequential(
            nn.Linear(condition_features, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
        )
        self.condition_bias = nn.Sequential(
            nn.Linear(condition_features, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )
        self.process_head = nn.Linear(hidden, process_classes)
        self.logit_scale = nn.Parameter(torch.tensor(1.0))

    def encode_event(
        self,
        nodes: torch.Tensor,
        mask: torch.Tensor,
        eta: torch.Tensor,
        phi: torch.Tensor,
        globals_: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self.encoder(nodes) * mask[..., None]
        for block in self.blocks:
            hidden = block(hidden, eta, phi, mask)
        count = mask.sum(dim=1, keepdim=True).clamp(min=1)
        mean = hidden.sum(dim=1) / count
        maximum = hidden.masked_fill(~mask[..., None], -1.0e9).max(dim=1).values
        maximum = torch.where(
            torch.isfinite(maximum), maximum, torch.zeros_like(maximum)
        )
        return self.event_head(torch.cat((mean, maximum, globals_), dim=-1))

    def conditioned_logits(
        self,
        event_embedding: torch.Tensor,
        conditions: torch.Tensor,
    ) -> torch.Tensor:
        query = self.event_query(event_embedding)
        key = self.condition_key(conditions)
        scale = self.logit_scale.exp().clamp(max=20.0) / (self.hidden ** 0.5)
        if conditions.ndim == 2:
            interaction = torch.sum(query * key, dim=-1)
        elif conditions.ndim == 3:
            interaction = torch.sum(query[:, None, :] * key, dim=-1)
        else:
            raise ValueError("conditions must have rank two or three")
        event_bias = self.event_bias(event_embedding).squeeze(-1)
        condition_bias = self.condition_bias(conditions).squeeze(-1)
        if conditions.ndim == 3:
            event_bias = event_bias[:, None]
        return scale * interaction + event_bias + condition_bias

    def grid_logits(
        self,
        event_embedding: torch.Tensor,
        condition_grid: torch.Tensor,
    ) -> torch.Tensor:
        """Score every event against every row of a common hypothesis grid."""
        if condition_grid.ndim != 2:
            raise ValueError("condition_grid must have shape [hypotheses, features]")
        query = self.event_query(event_embedding)
        key = self.condition_key(condition_grid)
        scale = self.logit_scale.exp().clamp(max=20.0) / (self.hidden ** 0.5)
        return (
            scale * torch.matmul(query, key.transpose(0, 1))
            + self.event_bias(event_embedding)
            + self.condition_bias(condition_grid).transpose(0, 1)
        )

    def forward(
        self,
        nodes: torch.Tensor,
        mask: torch.Tensor,
        eta: torch.Tensor,
        phi: torch.Tensor,
        globals_: torch.Tensor,
        conditions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        event_embedding = self.encode_event(nodes, mask, eta, phi, globals_)
        return (
            self.conditioned_logits(event_embedding, conditions),
            self.process_head(event_embedding),
        )


class MassConditionedProcessClassifier(nn.Module):
    """Mass-conditioned exhaustive process classifier for DNN/GNN/Transformer.

    Class zero is the signal hypothesis supplied through ``conditions``.  The
    remaining logits describe explicit background families and are independent
    of the tested signal mass.  This lets the final discriminant compare the
    hypothesis-specific signal logit directly with a physics-prior-weighted
    log-sum-exp of the background logits.
    """

    def __init__(
        self,
        *,
        architecture: str,
        max_jets: int = 10,
        node_features: int = 6,
        global_features: int = 30,
        condition_features: int = 6,
        hidden: int = 64,
        message_layers: int = 3,
        transformer_heads: int = 4,
        transformer_layers: int = 2,
        dropout: float = 0.12,
        process_classes: int = 6,
    ) -> None:
        super().__init__()
        if architecture not in {"dnn", "gnn", "transformer"}:
            raise ValueError(f"unsupported architecture: {architecture}")
        if process_classes < 2:
            raise ValueError("process_classes must include signal and a background")
        if architecture == "transformer" and hidden % transformer_heads:
            raise ValueError("hidden size must be divisible by transformer_heads")
        self.architecture = architecture
        self.hidden = hidden
        self.max_jets = max_jets

        if architecture == "dnn":
            width = max_jets * node_features + global_features
            self.dnn_encoder = nn.Sequential(
                nn.Linear(width, 4 * hidden),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(4 * hidden, 2 * hidden),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(2 * hidden, hidden),
                nn.SiLU(),
                nn.LayerNorm(hidden),
            )
        elif architecture == "gnn":
            self.node_encoder = nn.Sequential(
                nn.Linear(node_features, hidden),
                nn.SiLU(),
                nn.Linear(hidden, hidden),
                nn.LayerNorm(hidden),
            )
            self.message_blocks = nn.ModuleList(
                EdgeMessageBlock(hidden) for _ in range(message_layers)
            )
            self.gnn_event_head = nn.Sequential(
                nn.Linear(2 * hidden + global_features, 2 * hidden),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(2 * hidden, hidden),
                nn.SiLU(),
                nn.LayerNorm(hidden),
            )
        else:
            self.jet_projection = nn.Sequential(
                nn.Linear(node_features, hidden),
                nn.LayerNorm(hidden),
            )
            self.global_projection = nn.Sequential(
                nn.Linear(global_features, hidden),
                nn.SiLU(),
                nn.Linear(hidden, hidden),
                nn.LayerNorm(hidden),
            )
            layer = nn.TransformerEncoderLayer(
                d_model=hidden,
                nhead=transformer_heads,
                dim_feedforward=4 * hidden,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.transformer_encoder = nn.TransformerEncoder(
                layer,
                num_layers=transformer_layers,
                enable_nested_tensor=False,
            )
            self.transformer_event_head = nn.Sequential(
                nn.LayerNorm(hidden),
                nn.Linear(hidden, hidden),
                nn.SiLU(),
                nn.LayerNorm(hidden),
            )

        self.signal_event_query = nn.Linear(hidden, hidden)
        self.signal_event_bias = nn.Linear(hidden, 1)
        self.condition_key = nn.Sequential(
            nn.Linear(condition_features, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
        )
        self.condition_bias = nn.Sequential(
            nn.Linear(condition_features, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )
        self.background_head = nn.Linear(hidden, process_classes - 1)
        self.logit_scale = nn.Parameter(torch.tensor(1.0))

    def encode_event(
        self,
        nodes: torch.Tensor,
        mask: torch.Tensor,
        eta: torch.Tensor,
        phi: torch.Tensor,
        globals_: torch.Tensor,
    ) -> torch.Tensor:
        if self.architecture == "dnn":
            del mask, eta, phi
            return self.dnn_encoder(
                torch.cat((nodes.flatten(start_dim=1), globals_), dim=-1)
            )
        if self.architecture == "gnn":
            hidden = self.node_encoder(nodes) * mask[..., None]
            for block in self.message_blocks:
                hidden = block(hidden, eta, phi, mask)
            count = mask.sum(dim=1, keepdim=True).clamp(min=1)
            mean = hidden.sum(dim=1) / count
            maximum = hidden.masked_fill(~mask[..., None], -1.0e9).max(dim=1).values
            maximum = torch.where(
                torch.isfinite(maximum), maximum, torch.zeros_like(maximum)
            )
            return self.gnn_event_head(torch.cat((mean, maximum, globals_), dim=-1))

        del eta, phi
        jet_tokens = self.jet_projection(nodes)
        global_token = self.global_projection(globals_)[:, None, :]
        tokens = torch.cat((global_token, jet_tokens), dim=1)
        padding_mask = torch.cat(
            (
                torch.zeros((len(mask), 1), dtype=torch.bool, device=mask.device),
                ~mask,
            ),
            dim=1,
        )
        encoded = self.transformer_encoder(
            tokens, src_key_padding_mask=padding_mask
        )
        return self.transformer_event_head(encoded[:, 0])

    def signal_logits(
        self,
        event_embedding: torch.Tensor,
        conditions: torch.Tensor,
    ) -> torch.Tensor:
        query = self.signal_event_query(event_embedding)
        key = self.condition_key(conditions)
        scale = self.logit_scale.exp().clamp(max=20.0) / (self.hidden ** 0.5)
        if conditions.ndim == 2:
            interaction = torch.sum(query * key, dim=-1)
            event_bias = self.signal_event_bias(event_embedding).squeeze(-1)
        elif conditions.ndim == 3:
            interaction = torch.sum(query[:, None, :] * key, dim=-1)
            event_bias = self.signal_event_bias(event_embedding)
        else:
            raise ValueError("conditions must have rank two or three")
        return (
            scale * interaction
            + event_bias
            + self.condition_bias(conditions).squeeze(-1)
        )

    def class_logits(
        self,
        event_embedding: torch.Tensor,
        conditions: torch.Tensor,
    ) -> torch.Tensor:
        signal = self.signal_logits(event_embedding, conditions)
        background = self.background_head(event_embedding)
        if conditions.ndim == 2:
            return torch.cat((signal[:, None], background), dim=-1)
        expanded = background[:, None, :].expand(
            -1, conditions.shape[1], -1
        )
        return torch.cat((signal[..., None], expanded), dim=-1)

    def grid_logits(
        self,
        event_embedding: torch.Tensor,
        condition_grid: torch.Tensor,
    ) -> torch.Tensor:
        if condition_grid.ndim != 2:
            raise ValueError("condition_grid must have shape [hypotheses, features]")
        query = self.signal_event_query(event_embedding)
        key = self.condition_key(condition_grid)
        scale = self.logit_scale.exp().clamp(max=20.0) / (self.hidden ** 0.5)
        signal = (
            scale * torch.matmul(query, key.transpose(0, 1))
            + self.signal_event_bias(event_embedding)
            + self.condition_bias(condition_grid).transpose(0, 1)
        )
        background = self.background_head(event_embedding)[:, None, :].expand(
            -1, len(condition_grid), -1
        )
        return torch.cat((signal[..., None], background), dim=-1)

    def forward(
        self,
        nodes: torch.Tensor,
        mask: torch.Tensor,
        eta: torch.Tensor,
        phi: torch.Tensor,
        globals_: torch.Tensor,
        conditions: torch.Tensor,
    ) -> torch.Tensor:
        embedding = self.encode_event(nodes, mask, eta, phi, globals_)
        return self.class_logits(embedding, conditions)


class GlobalOnlyClassifier(nn.Module):
    """Small control model used to test whether the graph adds information."""

    def __init__(self, global_features: int = 9, hidden: int = 32) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(global_features, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def forward(
        self,
        nodes: torch.Tensor,
        mask: torch.Tensor,
        eta: torch.Tensor,
        phi: torch.Tensor,
        globals_: torch.Tensor,
    ) -> torch.Tensor:
        del nodes, mask, eta, phi
        return self.network(globals_).squeeze(-1)


class FlattenDNNClassifier(nn.Module):
    """Ordered-jet DNN baseline; jets are supplied in descending corrected pT."""

    def __init__(
        self,
        max_jets: int = 10,
        node_features: int = 6,
        global_features: int = 9,
        hidden: int = 32,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        width = max_jets * node_features + global_features
        self.network = nn.Sequential(
            nn.Linear(width, 4 * hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(4 * hidden, 2 * hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(2 * hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def forward(
        self,
        nodes: torch.Tensor,
        mask: torch.Tensor,
        eta: torch.Tensor,
        phi: torch.Tensor,
        globals_: torch.Tensor,
    ) -> torch.Tensor:
        del mask, eta, phi
        flat = nodes.flatten(start_dim=1)
        return self.network(torch.cat((flat, globals_), dim=-1)).squeeze(-1)


class JetTransformerClassifier(nn.Module):
    """Set transformer over jet tokens with a global/mass hypothesis token."""

    def __init__(
        self,
        node_features: int = 6,
        global_features: int = 9,
        hidden: int = 32,
        heads: int = 4,
        layers: int = 2,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        if hidden % heads:
            raise ValueError("hidden size must be divisible by the number of heads")
        self.jet_projection = nn.Sequential(
            nn.Linear(node_features, hidden),
            nn.LayerNorm(hidden),
        )
        self.global_projection = nn.Sequential(
            nn.Linear(global_features, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=heads,
            dim_feedforward=4 * hidden,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=layers, enable_nested_tensor=False
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def forward(
        self,
        nodes: torch.Tensor,
        mask: torch.Tensor,
        eta: torch.Tensor,
        phi: torch.Tensor,
        globals_: torch.Tensor,
    ) -> torch.Tensor:
        del eta, phi
        jet_tokens = self.jet_projection(nodes)
        global_token = self.global_projection(globals_)[:, None, :]
        tokens = torch.cat((global_token, jet_tokens), dim=1)
        padding_mask = torch.cat(
            (
                torch.zeros((len(mask), 1), dtype=torch.bool, device=mask.device),
                ~mask,
            ),
            dim=1,
        )
        encoded = self.encoder(tokens, src_key_padding_mask=padding_mask)
        return self.head(encoded[:, 0]).squeeze(-1)
