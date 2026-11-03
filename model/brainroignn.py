"""BrainRoiGNN: edge-aware graph encoding, fixed anatomical ROI pooling,
hemispheric asymmetry tokens and an ROI-level Transformer.

Input per epoch: a complete 60-node graph with node features (60 x 10,
log-relative band power) and edge features (10 wPLI bands per edge).
"""
from __future__ import annotations

from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv
from torch_geometric.utils import to_dense_batch

ROI_NAMES = ["Frontal-L", "Frontal-R", "Sensorimotor-L", "Sensorimotor-R",
             "Posterolateral-L", "Posterolateral-R", "Anterior-Midline", "Posterior-Midline"]
ROI_CHANNELS = [
    {"Fp1", "AF7", "AF3", "F7", "F5", "F3", "F1"},
    {"Fp2", "AF4", "AF8", "F2", "F4", "F6", "F8"},
    {"FC5", "FC3", "FC1", "C5", "C3", "C1", "CP5", "CP3", "CP1"},
    {"FC2", "FC4", "FC6", "C2", "C4", "C6", "CP2", "CP4", "CP6"},
    {"FT7", "T7", "TP9", "TP7", "P7", "P5", "P3", "P1", "PO7", "O1"},
    {"FT8", "FT10", "T8", "TP8", "TP10", "P2", "P4", "P6", "P8", "PO8", "O2"},
    {"AFz", "Fz", "FCz"},
    {"Cz", "CPz", "POz", "Oz"},
]
LR_PAIRS = [(0, 1), (2, 3), (4, 5)]          # frontal, sensorimotor, posterolateral (left, right)


def channel_to_roi(ch_names: List[str]) -> np.ndarray:
    lut = {c: r for r, chs in enumerate(ROI_CHANNELS) for c in chs}
    return np.array([lut[c] for c in ch_names], dtype=np.int64)


class BrainRoiGNN(nn.Module):
    def __init__(self, node_input_dim: int, edge_input_dim: int, ch_to_roi, n_rois: int = 8,
                 lr_pairs=LR_PAIRS, hidden_dim: int = 64, num_blocks: int = 3, num_heads: int = 4,
                 dropout: float = 0.4, n_classes: int = 2):
        super().__init__()
        self.lr_pairs = list(lr_pairs)
        # Fixed (non-learned) row-normalised ROI assignment matrix (n_rois x n_channels).
        ch_to_roi_t = torch.as_tensor(ch_to_roi, dtype=torch.long)
        self.register_buffer("ch_to_roi", ch_to_roi_t)
        assign = torch.zeros(n_rois, ch_to_roi_t.numel())
        for i, r in enumerate(ch_to_roi_t.tolist()):
            assign[r, i] = 1.0
        self.register_buffer("roi_assign", assign / assign.sum(dim=1, keepdim=True).clamp_min(1.0))

        # Edge-aware graph encoding: linear encoders, then GINE layers.
        # Parameter creation order is kept fixed so that seeded runs reproduce exactly.
        self.node_encoder = nn.Linear(node_input_dim, hidden_dim)
        self.edge_encoder = nn.Linear(edge_input_dim, hidden_dim)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_blocks):
            mlp = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
            self.convs.append(GINEConv(mlp, train_eps=True))
            self.norms.append(nn.BatchNorm1d(hidden_dim))
        self.dropout = nn.Dropout(dropout)

        # 8 ROI tokens + 3 asymmetry tokens, each with a learnable token embedding.
        self.n_tokens = n_rois + len(self.lr_pairs)
        self.roi_pos = nn.Parameter(torch.randn(self.n_tokens, hidden_dim) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads,
                                               dim_feedforward=hidden_dim * 2, dropout=dropout,
                                               batch_first=True, activation="gelu")
        self.roi_encoder = nn.TransformerEncoder(enc_layer, num_layers=2)
        self.classifier = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(),
                                        nn.Dropout(dropout), nn.Linear(hidden_dim, n_classes))

    def tokens(self, data) -> torch.Tensor:
        """Channel encoding -> (B, 11, hidden) ROI and asymmetry tokens."""
        x = self.node_encoder(data.x)
        ea = self.edge_encoder(data.edge_attr)
        for conv, norm in zip(self.convs, self.norms):
            x = self.dropout(F.relu(norm(conv(x, data.edge_index, ea))))
        n_static = getattr(data, "static_num_nodes", None)
        if n_static is not None:            # every graph has the same nodes, in order
            x_dense = x.view(-1, n_static, x.shape[-1])
        else:
            x_dense, _ = to_dense_batch(x, data.batch)
        roi = torch.einsum("rn,bnh->brh", self.roi_assign, x_dense)
        asym = torch.stack([roi[:, l] - roi[:, r] for (l, r) in self.lr_pairs], dim=1)
        return torch.cat([roi, asym], dim=1)

    def forward(self, data) -> torch.Tensor:
        t = self.roi_encoder(self.tokens(data) + self.roi_pos.unsqueeze(0))
        return self.classifier(torch.cat([t.mean(dim=1), t.max(dim=1).values], dim=-1))
