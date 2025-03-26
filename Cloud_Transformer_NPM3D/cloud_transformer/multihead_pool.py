import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from cloud_transformer.core import Splat, DifferentiablePositions
from cloud_transformer.utils_ct import PlaneTransformer, VolTransformer
from cloud_transformer.improvements import SplatSoftmaxAggregator

class MultiHeadPool(nn.Module):
    def __init__(self,
                 model_dim,
                 feature_dim,
                 grid_size,
                 grid_dim,
                 num_heads,
                 scales=False,
                 use_checkpoint=False,
                 use_softmax=False):
        super().__init__()

        self.feature_dim = feature_dim
        self.model_dim = model_dim
        self.grid_size = grid_size
        self.grid_dim = grid_dim
        self.num_heads = num_heads
        self.use_checkpoint = use_checkpoint
        self.use_softmax = use_softmax

        self.keys_values_pred = nn.Conv1d(
            self.model_dim,
            self.num_heads * (self.feature_dim + 3),
            kernel_size=1,
            bias=False
        )

        self.values_bn = nn.BatchNorm1d(self.num_heads * self.feature_dim)
        self.key_bn = nn.BatchNorm1d(self.num_heads * 3)
        nn.init.zeros_(self.key_bn.weight)

        self.diff_pos_generator = DifferentiablePositions(
            tensor_size=self.grid_size,
            dim=self.grid_dim,
            heads=self.num_heads
        )

        self.splat = (
            SplatSoftmaxAggregator(tensor_size=self.grid_size,
                                    dim=self.grid_dim,
                                    heads=self.num_heads,
                                    init_temperature=1.0)
            if self.use_softmax else
            Splat(tensor_size=self.grid_size,
                  dim=self.grid_dim,
                  heads=self.num_heads)
        )

        self.transform = VolTransformer(self.num_heads, scales=scales) if self.grid_dim == 3 else PlaneTransformer(self.num_heads, scales=scales)

    def _forward_body(self, input_tensor, original_points):
        key_values = self.keys_values_pred(input_tensor)
        keys_offset = self.key_bn(key_values[:, :self.num_heads * 3])
        values = self.values_bn(key_values[:, self.num_heads * 3:])

        keys_offset = keys_offset.view(input_tensor.shape[0], self.num_heads, 3, input_tensor.shape[-1])
        transformed_keys = self.transform(original_points.unsqueeze(1) + keys_offset)
        keys = transformed_keys.reshape(input_tensor.shape[0], self.num_heads * self.grid_dim, input_tensor.shape[-1])
        lattice = torch.tanh(keys)

        local_coord, flattened_index = self.diff_pos_generator(lattice)
        splat_out = self.splat(local_coord, flattened_index, values)
        return splat_out

    def forward(self, input_tensor, original_points):
        if self.use_checkpoint and self.training:
            result = checkpoint(self._forward_body, input_tensor, original_points, use_reentrant=False)
        else:
            result = self._forward_body(input_tensor, original_points)
        return result
