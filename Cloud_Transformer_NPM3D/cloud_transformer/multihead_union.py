import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from cloud_transformer.core import Splat, Slice, DifferentiablePositions
from cloud_transformer.utils_ct import PlaneTransformer, VolTransformer
from cloud_transformer.improvements import SplatSoftmaxAggregator

class MultiHeadAttention(nn.Module):
    def __init__(self,
                 input_dim,
                 attention_feature_dim,
                 output_dim,
                 grid_size,
                 grid_dim,
                 num_heads,
                 use_scales=False,
                 use_checkpoint=False,
                 use_softmax=False):
        super().__init__()
        assert grid_dim in (2, 3), "grid_dim must be either 2 or 3."

        self.use_checkpoint = use_checkpoint
        self.use_softmax = use_softmax
        self.num_heads = num_heads
        self.grid_dim = grid_dim

        self.kv_conv = nn.Conv1d(input_dim,
                                 num_heads * (attention_feature_dim + 3),
                                 kernel_size=1,
                                 bias=False)
        self.keys_bn = nn.BatchNorm1d(num_heads * 3)
        self.values_bn = nn.BatchNorm1d(num_heads * attention_feature_dim)
        nn.init.zeros_(self.keys_bn.weight)

        self.diff_pos = DifferentiablePositions(tensor_size=grid_size,
                                                dim=grid_dim,
                                                heads=num_heads)
        self.splat = (SplatSoftmaxAggregator(tensor_size=grid_size,
                                             dim=grid_dim,
                                             heads=num_heads,
                                             init_temperature=1.0)
                      if use_softmax else
                      Splat(tensor_size=grid_size,
                            dim=grid_dim,
                            heads=num_heads))
        self.slice = Slice(tensor_size=grid_size,
                           dim=grid_dim,
                           heads=num_heads)

        Conv = nn.Conv3d if grid_dim == 3 else nn.Conv2d
        self.conv = Conv(num_heads * attention_feature_dim,
                         num_heads * attention_feature_dim,
                         kernel_size=3,
                         padding=1,
                         groups=num_heads,
                         bias=True)
        self.after = nn.Sequential(nn.BatchNorm1d(num_heads * attention_feature_dim),
                                   nn.ReLU(inplace=True))
        self.transform = VolTransformer(num_heads, scales=use_scales) if grid_dim == 3 else PlaneTransformer(num_heads, scales=use_scales)

    def _forward_body(self, x, orig_points, pts_padding=None):
        kv = self.kv_conv(x)
        keys_offset = self.keys_bn(kv[:, :self.num_heads * 3])
        values = self.values_bn(kv[:, self.num_heads * 3:])

        keys_offset = keys_offset.view(x.size(0), self.num_heads, 3, x.size(-1))
        transformed_keys = self.transform(orig_points.unsqueeze(1) + keys_offset)
        keys = transformed_keys.reshape(x.size(0), self.num_heads * self.grid_dim, x.size(-1))
        lattice = torch.tanh(keys)

        local_coord, flat_idx = self.diff_pos(lattice)
        splat_out = self.splat(local_coord, flat_idx, values, pts_padding)
        conv_out = self.conv(splat_out)
        sliced = self.slice(local_coord, flat_idx, conv_out, pts_padding)
        return self.after(sliced)

    def forward(self, x, orig_points):
        if isinstance(orig_points, tuple):
            orig_points, pts_padding = orig_points
        else:
            pts_padding = None
        if self.use_checkpoint and self.training:
            return checkpoint(self._forward_body, x, orig_points, pts_padding, use_reentrant=False)
        return self._forward_body(x, orig_points, pts_padding)

class MultiHeadUnionAttention(nn.Module):
    def __init__(self,
                 input_dim,
                 feature_dims,
                 grid_sizes,
                 grid_dims,
                 num_heads_list,
                 output_dim=None,
                 use_scales=False,
                 use_checkpoint=False,
                 use_softmax=False):
        super().__init__()
        self.output_dim = output_dim or input_dim

        total_features = sum([h * f for h, f in zip(num_heads_list, feature_dims)])
        self.after = nn.Sequential(
            nn.Conv1d(total_features, self.output_dim, kernel_size=1, bias=False),
            nn.BatchNorm1d(self.output_dim),
            nn.ReLU(inplace=True)
        )
        self.shortcut = nn.Identity() if self.output_dim == input_dim else nn.Conv1d(input_dim, self.output_dim, kernel_size=1, bias=False)

        self.attentions = nn.ModuleList([
            MultiHeadAttention(input_dim=input_dim,
                               attention_feature_dim=feat_dim,
                               output_dim=self.output_dim,
                               grid_size=grid_size,
                               grid_dim=grid_dim,
                               num_heads=num_heads,
                               use_scales=use_scales,
                               use_checkpoint=use_checkpoint,
                               use_softmax=use_softmax)
            for feat_dim, grid_size, grid_dim, num_heads in zip(feature_dims, grid_sizes, grid_dims, num_heads_list)
        ])

    def forward(self, x, orig_points):
        residual = self.shortcut(x)
        outputs = [att(x, orig_points) for att in self.attentions]
        concat = torch.cat(outputs, dim=1)
        return residual + self.after(concat)
