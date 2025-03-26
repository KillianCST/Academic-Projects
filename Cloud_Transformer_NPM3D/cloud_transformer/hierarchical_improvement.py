from cloud_transformer.multihead_union import MultiHeadUnionAttention
import torch
import torch.nn as nn


class GatedFusion(nn.Module):
    def __init__(self, channels, num_scales):
        super().__init__()
        self.gates = nn.Parameter(torch.zeros(num_scales, channels))

    def forward(self, features):
        # features: List of tensors [B, C, P]
        stacked = torch.stack(features, dim=0)  # [num_scales, B, C, P]
        gates = torch.softmax(self.gates, dim=0).unsqueeze(1).unsqueeze(-1)  # [num_scales, 1, C, 1]
        gated_features = (stacked * gates).sum(dim=0)  # [B, C, P]
        return gated_features


class HierarchicalMultiHeadAttention(nn.Module):
    def __init__(self, input_dim, heads, use_scales=True, use_checkpoint=False, use_softmax=False):
        super().__init__()

        self.attn_high_res = MultiHeadUnionAttention(
            input_dim=input_dim,
            feature_dims=[4, 4],
            num_heads_list=[heads, heads],
            grid_sizes=[128, 32],  # High resolution
            grid_dims=[2, 3],
            output_dim=input_dim,
            use_scales=use_scales,
            use_checkpoint=use_checkpoint,
            use_softmax=use_softmax,
        )

        self.attn_mid_res = MultiHeadUnionAttention(
            input_dim=input_dim,
            feature_dims=[16, 16],
            num_heads_list=[heads, heads],
            grid_sizes=[64, 16],  # Medium resolution
            grid_dims=[2, 3],
            output_dim=input_dim,
            use_scales=use_scales,
            use_checkpoint=use_checkpoint,
            use_softmax=use_softmax,
        )

        self.attn_low_res = MultiHeadUnionAttention(
            input_dim=input_dim,
            feature_dims=[16, 32],
            num_heads_list=[heads, heads],
            grid_sizes=[32, 8],  # Low resolution
            grid_dims=[2, 3],
            output_dim=input_dim,
            use_scales=use_scales,
            use_checkpoint=use_checkpoint,
            use_softmax=use_softmax,
        )

        self.attn_very_low_res = MultiHeadUnionAttention(
            input_dim=input_dim,
            feature_dims=[16, 32],
            num_heads_list=[heads, heads],
            grid_sizes=[16, 4],  # very Low resolution
            grid_dims=[2, 3],
            output_dim=input_dim,
            use_scales=use_scales,
            use_checkpoint=use_checkpoint,
            use_softmax=use_softmax,
        )

        self.gated_fusion = GatedFusion(input_dim, num_scales=4)


    def forward(self, x, orig):
        high_res = self.attn_high_res(x, orig)
        mid_res = self.attn_mid_res(x, orig)
        low_res = self.attn_low_res(x, orig)
        very_low_res = self.attn_very_low_res(x, orig)

        return self.gated_fusion([high_res, mid_res, low_res, very_low_res])

    
    

class MultiInputSequential(nn.Sequential):
    def forward(self, *inputs):
        for module in self:
            if isinstance(inputs, tuple):
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs
    

