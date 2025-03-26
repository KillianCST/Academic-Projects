import torch
import torch.nn as nn
import numpy as np
from torch_scatter import scatter_softmax, scatter_add
from cloud_transformer.core import DifferentiableGridModule


class SplatSoftmaxAggregator(DifferentiableGridModule):
    def __init__(self, tensor_size=20, heads=4, dim=3, init_temperature=1.0):
        super().__init__(tensor_size, heads, dim)
        # One temperature per head
        self.temperature = nn.Parameter(torch.full((heads, 1), init_temperature))

    def forward(self, local_coordinate, flattened_index, features, pts_padding=None):
        B, H = features.size(0), self.heads
        feature_dim = features.size(1) // H
        V, P = local_coordinate.size(2), local_coordinate.size(3)
        
        # Flatten coords & indices: [B, H, V*P]
        logits = (local_coordinate.view(B, H, -1) * self.temperature.view(1, H, 1))
        index_flat = flattened_index.view(B, H, -1)

        # Mask padded contributions before softmax
        if pts_padding is not None:
            mask = pts_padding.view(B, 1, 1, P).expand(B, H, V, P).reshape(B, H, -1)
            logits = logits.masked_fill(mask == 0, float('-inf'))

        # Compute normalized weights per grid‑cell
        weights = scatter_softmax(logits, index_flat, dim=-1)

        # Reshape features → [B, H, feature_dim, V*P]
        feats = features.view(B, H, feature_dim, P)
        feats = feats.unsqueeze(3).expand(-1, -1, -1, V, -1).reshape(B, H, feature_dim, -1)

        # Weighted aggregation into grid
        weighted = feats * weights.unsqueeze(2)
        grid_total = int(np.prod(self.tensor_size))
        out = torch.zeros(B, H, feature_dim, grid_total, device=features.device)
        idx = index_flat.unsqueeze(2).expand(-1, -1, feature_dim, -1)
        aggregated = scatter_add(weighted, idx, dim=-1, out=out)

        # Reshape to final grid
        shape = (*self.tensor_size,) if self.dim == 2 else (*self.tensor_size, )
        return aggregated.view(B, H * feature_dim, *shape)


class ScaleNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(dim ** 0.5))
        self.eps = eps

    def forward(self, x):
        norm = torch.norm(x, dim=1, keepdim=True).clamp(min=self.eps)
        return self.scale * x / norm


class ChannelLayerNorm(nn.Module):
    def __init__(self, channels, eps=1e-5):
        super().__init__()
        self.layer_norm = nn.LayerNorm(channels, eps=eps)

    def forward(self, x):
        if x.dim() == 3:
            x = x.permute(0, 2, 1)  # [B, C, P] → [B, P, C]
            x = self.layer_norm(x)
            x = x.permute(0, 2, 1)  # [B, P, C] → [B, C, P]
        elif x.dim() == 4:
            x = x.permute(0, 2, 3, 1)  # [B, C, H, W] → [B, H, W, C]
            x = self.layer_norm(x)
            x = x.permute(0, 3, 1, 2)  # [B, H, W, C] → [B, C, H, W]
        elif x.dim() == 5:
            x = x.permute(0, 2, 3, 4, 1)  # [B, C, D, H, W] → [B, D, H, W, C]
            x = self.layer_norm(x)
            x = x.permute(0, 4, 1, 2, 3)  # [B, D, H, W, C] → [B, C, D, H, W]
        else:
            raise ValueError(f'Unsupported input dimensions: {x.dim()}')
        return x