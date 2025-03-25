import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from cloud_transformer.core import Splat, Slice, DifferentiablePositions
from cloud_transformer.utils_ct import PlaneTransformer, VolTransformer, AdaIn1dUpd

def _apply_style(module: nn.Sequential, x: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
    """Apply style-conditioned layers in a sequential module."""
    for layer in module:
        if isinstance(layer, AdaIn1dUpd):
            x = layer(x, style)
        else:
            x = layer(x)
    return x

class MultiHeadAdaIn(nn.Module):
    def __init__(self,
                 input_dim: int,
                 attention_feature_dim: int,
                 output_dim: int,
                 grid_size: int,
                 grid_dim: int,
                 num_heads: int,
                 n_latent: int = 256,
                 use_scales: bool = False,
                 use_checkpoint: bool = False):
        super().__init__()
        assert grid_dim in (2, 3), "grid_dim must be 2 or 3"
        self.input_dim = input_dim
        self.attention_feature_dim = attention_feature_dim
        self.output_dim = output_dim
        self.grid_size = grid_size
        self.grid_dim = grid_dim
        self.num_heads = num_heads
        self.n_latent = n_latent
        self.use_checkpoint = use_checkpoint

        self.scale = nn.Parameter(torch.tensor(0.0), requires_grad=True)

        self.kv_conv = nn.Conv1d(input_dim,
                                 num_heads * (attention_feature_dim + 3),
                                 kernel_size=1, bias=False)
        self.keys_bn = AdaIn1dUpd(num_heads * 3, num_latent=n_latent)
        self.values_bn = AdaIn1dUpd(num_heads * attention_feature_dim, num_latent=n_latent)

        self.diff_pos = DifferentiablePositions(tensor_size=grid_size,
                                                dim=grid_dim,
                                                heads=num_heads)
        self.splat = Splat(tensor_size=grid_size, dim=grid_dim, heads=num_heads)
        self.slice = Slice(tensor_size=grid_size, dim=grid_dim, heads=num_heads)

        conv_layer = nn.Conv3d if grid_dim == 3 else nn.Conv2d
        self.conv = nn.Sequential(
            conv_layer(num_heads * attention_feature_dim,
                       num_heads * attention_feature_dim,
                       kernel_size=3, padding=1, groups=num_heads, bias=True)
        )
        self.after = nn.Sequential(
            AdaIn1dUpd(num_heads * attention_feature_dim, num_latent=n_latent),
            nn.ReLU(inplace=True)
        )
        self.transform = VolTransformer(num_heads, scales=use_scales) if grid_dim == 3 else PlaneTransformer(num_heads, scales=use_scales)


    def _forward_body(self, x: torch.Tensor, orig_points: torch.Tensor, style: torch.Tensor):
        kv = self.kv_conv(x)
        keys_offset = self.keys_bn(kv[:, :self.num_heads * 3], style)
        values = self.values_bn(kv[:, self.num_heads * 3:], style)

        keys_offset = keys_offset.view(x.shape[0], self.num_heads, 3, x.shape[-1])
        transformed_keys = self.transform(orig_points[:, None] + self.scale * keys_offset)
        keys = transformed_keys.reshape(x.shape[0], self.num_heads * self.grid_dim, x.shape[-1])
        lattice = torch.tanh(keys)

        local_coord, flattened_index = self.diff_pos(lattice)
        splat_out = self.splat(local_coord, flattened_index, values)
        conv_out = self.conv(splat_out)
        sliced = self.slice(local_coord, flattened_index, conv_out)
        result = _apply_style(self.after, sliced, style)

        with torch.no_grad():
            occupancy = (torch.abs(splat_out) > 1e-9).sum().float() / (keys.size(0) * self.attention_feature_dim * self.num_heads)
            stats = (occupancy, torch.mean(keys).detach(), torch.var(keys).detach(), None)
        return result, stats, lattice

    def forward(self, x: torch.Tensor, orig_points: torch.Tensor, style: torch.Tensor, return_lattice: bool = False):
        if self.use_checkpoint and self.training:
            result, stats, lattice = checkpoint(self._forward_body, x, orig_points, style, use_reentrant=False)
        else:
            result, stats, lattice = self._forward_body(x, orig_points, style)
        return (result, lattice) if return_lattice else (result, stats)

class MultiHeadUnionAdaIn(nn.Module):
    def __init__(self,
                 input_dim: int,
                 feature_dims: list,
                 grid_sizes: list,
                 grid_dims: list,
                 num_heads_list: list,
                 output_dim: int = None,
                 n_latent: int = 256,
                 use_scales: bool = False,
                 use_checkpoint: bool = False):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim or input_dim
        total_features = sum([h * f for h, f in zip(num_heads_list, feature_dims)])
        self.shortcut = nn.Sequential()
        if self.input_dim != self.output_dim:
            self.shortcut.add_module('conv', nn.Conv1d(input_dim, self.output_dim, kernel_size=1, bias=False))
            self.shortcut.add_module('adain', AdaIn1dUpd(self.output_dim, num_latent=n_latent))
        self.after = nn.Sequential(
            nn.Conv1d(total_features, self.output_dim, kernel_size=1, bias=False),
            AdaIn1dUpd(self.output_dim, num_latent=n_latent),
            nn.ReLU(inplace=True)
        )
        self.attentions = nn.ModuleList([
            MultiHeadAdaIn(input_dim, feat_dim, self.output_dim, grid_size, grid_dim, heads,
                           n_latent=n_latent, use_scales=use_scales, use_checkpoint=use_checkpoint)
            for feat_dim, grid_size, grid_dim, heads in zip(feature_dims, grid_sizes, grid_dims, num_heads_list)
        ])
     

    def forward(self, x: torch.Tensor, orig_points: torch.Tensor, style: torch.Tensor):
        residual = _apply_style(self.shortcut, x, style) if len(self.shortcut) > 0 else x
        outputs, stats_list = [], []
        for attention in self.attentions:
            out, stats = attention(x, orig_points, style)
            outputs.append(out)
            stats_list.append(stats)
        concat = torch.cat(outputs, dim=1)
        gathered = _apply_style(self.after, concat, style)
        return residual + gathered, stats_list
