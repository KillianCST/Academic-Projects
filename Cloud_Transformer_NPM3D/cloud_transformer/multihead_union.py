import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from torch.nn.init import xavier_uniform_
from cloud_transformer.core import Splat, Slice, DifferentiablePositions
from cloud_transformer.utils_ct import PlaneTransformer, VolTransformer

class MultiHeadAttention(nn.Module):
    def __init__(self,
                 input_dim,
                 attention_feature_dim,
                 output_dim,
                 grid_size,
                 grid_dim,
                 num_heads,
                 use_scales=False,
                 use_checkpoint=False):
        """
        Multi-head attention module with splatting, slicing, and grouped convolution.

        Args:
            input_dim (int): Number of input channels.
            attention_feature_dim (int): Feature dimension for each attention head.
            output_dim (int): Output model dimension.
            grid_size (int): Size of the tensor/grid.
            grid_dim (int): Dimensionality of the grid (2 or 3).
            num_heads (int): Number of attention heads.
            use_scales (bool): Whether to use scale transformation.
            use_checkpoint (bool): Whether to use checkpointing for memory efficiency.
        """
        super().__init__()
        assert grid_dim in (2, 3), "grid_dim must be either 2 or 3."

        self.input_dim = input_dim
        self.attention_feature_dim = attention_feature_dim
        self.output_dim = output_dim
        self.grid_size = grid_size
        self.grid_dim = grid_dim
        self.num_heads = num_heads
        self.use_checkpoint = use_checkpoint

        # Predict key offsets (first num_heads * 3 channels) and attention values (remaining channels)
        self.kv_conv = nn.Conv1d(self.input_dim,
                                 self.num_heads * (self.attention_feature_dim + 3),
                                 kernel_size=1,
                                 bias=False)

        # Batch normalization for keys and values separately
        self.values_bn = nn.BatchNorm1d(self.num_heads * self.attention_feature_dim)
        self.keys_bn = nn.BatchNorm1d(self.num_heads * 3)
        # Zero-initialize key batch norm weights to start with an identity-like mapping
        nn.init.zeros_(self.keys_bn.weight)

        # Differentiable positions generator to compute lattice coordinates
        self.diff_pos_generator = DifferentiablePositions(tensor_size=self.grid_size,
                                                          dim=self.grid_dim,
                                                          heads=self.num_heads)
        # Splatting layer to project attention values onto the lattice
        self.splat = Splat(tensor_size=self.grid_size,
                           dim=self.grid_dim,
                           heads=self.num_heads)
        # Slicing layer to retrieve features from the lattice
        self.slice = Slice(tensor_size=self.grid_size,
                           dim=self.grid_dim,
                           heads=self.num_heads)

        # Grouped convolution after splatting
        if self.grid_dim == 3:
            conv_layer = nn.Conv3d(self.num_heads * self.attention_feature_dim,
                                   self.num_heads * self.attention_feature_dim,
                                   kernel_size=3,
                                   stride=1,
                                   padding=1,
                                   groups=self.num_heads,
                                   bias=True)
        else:
            conv_layer = nn.Conv2d(self.num_heads * self.attention_feature_dim,
                                   self.num_heads * self.attention_feature_dim,
                                   kernel_size=3,
                                   stride=1,
                                   padding=1,
                                   groups=self.num_heads,
                                   bias=True)
        self.conv = nn.Sequential(conv_layer)

        # Post-processing: batch normalization and ReLU
        self.after = nn.Sequential(
            nn.BatchNorm1d(self.num_heads * self.attention_feature_dim),
            nn.ReLU(inplace=True)
        )

        # Transformation module: either volumetric or planar
        if self.grid_dim == 3:
            self.transform = VolTransformer(self.num_heads, scales=use_scales)
        else:
            self.transform = PlaneTransformer(self.num_heads, scales=use_scales)

        self._reset_parameters()

    def _reset_parameters(self):
        #xavier_uniform_(self.kv_conv.weight)
        #xavier_uniform_(self.conv[0].weight)
        torch.nn.init.zeros_(self.keys_bn.weight)

    def _forward_body(self, input_tensor, orig_points, pts_padding):
        """
        Heavy computation block for the forward pass.

        Args:
            input_tensor (Tensor): Input features of shape (B, C, P).
            orig_points (Tensor): Original point cloud of shape (B, 3, P).
            pts_padding (Tensor or None): Optional padding mask.

        Returns:
            keys (Tensor): Transformed keys of shape (B, num_heads * grid_dim, P).
            lattice (Tensor): Lattice coordinates (after tanh) of shape (B, num_heads * grid_dim, P).
            local_coord (Tensor): Local coordinates from the differentiable positions module.
            flattened_index (Tensor): Flattened indices corresponding to lattice positions.
            splat_out (Tensor): Output of the splatting operation.
            final_out (Tensor): Output after grouped convolution and slicing.
        """
        # Compute key offsets and attention values
        kv = self.kv_conv(input_tensor)  # (B, num_heads*(attention_feature_dim+3), P)
        keys_offset = self.keys_bn(kv[:, :self.num_heads * 3])
        # Correct slicing: values start after the first num_heads*3 channels
        values = self.values_bn(kv[:, self.num_heads * 3:])
        
        # Reshape keys_offset to (B, num_heads, 3, P) and add to original points
        keys_offset = keys_offset.view(input_tensor.shape[0], self.num_heads, 3, input_tensor.shape[-1])
        transformed_keys = self.transform(orig_points[:, None] + keys_offset)
        # Flatten transformed keys to (B, num_heads * grid_dim, P)
        keys = transformed_keys.reshape(input_tensor.shape[0], self.num_heads * self.grid_dim, input_tensor.shape[-1])
        lattice = torch.tanh(keys)
        
        # Compute lattice local coordinates and flattened indices
        local_coord, flattened_index = self.diff_pos_generator(lattice)
        # Splat the attention values onto the lattice
        splat_out = self.splat(local_coord, flattened_index, values, pts_padding)
        # Process with grouped convolution
        conv_out = self.conv(splat_out)
        # Slice the lattice.
        sliced = self.slice(local_coord, flattened_index, conv_out, pts_padding)
        final_out = self.after(sliced)
        return keys, lattice, splat_out, final_out

    def forward(self, input_tensor, orig_points, return_lattice=False):
        """
        Forward pass for multi-head attention.

        Args:
            input_tensor (Tensor): Input feature map of shape (B, C, P).
            orig_points (Tensor or tuple): Original point cloud (B, 3, P) or a tuple (orig_points, pts_padding).
            return_lattice (bool): If True, also return the computed lattice.

        Returns:
            result (Tensor or tuple): Final output features, or (features, lattice) if return_lattice is True.
            stats (tuple): Tuple containing occupancy, mean(keys), var(keys), and None.
        """
        if isinstance(orig_points, tuple):
            orig_points, pts_padding = orig_points
        else:
            pts_padding = None

        if self.use_checkpoint and self.training:
            def custom_forward(input_tensor, orig_points):
                return self._forward_body(input_tensor, orig_points, pts_padding)
            keys, lattice, splat_out, final_out = checkpoint(
                custom_forward, input_tensor, orig_points, use_reentrant=False)
        else:
            keys, lattice, splat_out, final_out = self._forward_body(
                input_tensor, orig_points, pts_padding)

        with torch.no_grad():
            occupancy = (torch.abs(splat_out) > 1e-9).sum().float() / (keys.size(0) * self.attention_feature_dim * self.num_heads)
            keys_mean = torch.mean(keys).detach()
            keys_var = torch.var(keys).detach()
            stats = (occupancy, keys_mean, keys_var, None)

        result = final_out
        if return_lattice:
            result = (result, lattice)
        return result, stats

class MultiHeadUnionAttention(nn.Module):
    def __init__(self,
                 input_dim,
                 features_dims,
                 grid_sizes,
                 grid_dims,
                 num_heads_list,
                 output_dim=None,
                 use_scales=False,
                 use_checkpoint=False):
        """
        Multi-branch union attention module that aggregates multiple MultiHeadAttention branches.

        Args:
            input_dim (int): Number of input channels.
            features_dims (list of int): Feature dimensions for each attention branch.
            grid_sizes (list of int): Grid sizes for each branch.
            grid_dims (list of int): Grid dimensionalities (2 or 3) for each branch.
            num_heads_list (list of int): Number of heads for each branch.
            output_dim (int, optional): Output model dimension. Defaults to input_dim.
            use_scales (bool): Whether to use scale transformation.
            use_checkpoint (bool): Whether to use checkpointing in each branch.
        """
        super().__init__()
        self.input_dim = input_dim
        self.features_dims = features_dims
        self.grid_sizes = grid_sizes
        self.grid_dims = grid_dims
        self.num_heads_list = num_heads_list
        assert len(self.features_dims) == len(self.grid_sizes) == len(self.grid_dims) == len(self.num_heads_list), \
            "Mismatch in the number of branches for features_dims, grid_sizes, grid_dims, and num_heads_list."

        self.output_dim = output_dim if output_dim is not None else self.input_dim

        # Calculate total features after concatenation
        total_features = sum([heads * feat_dim for heads, feat_dim in zip(self.num_heads_list, self.features_dims)])
        self.after = nn.Sequential(
            nn.Conv1d(total_features, self.output_dim, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm1d(self.output_dim),
            nn.ReLU(inplace=True)
        )

        # Shortcut connection for residual fusion
        self.shortcut = nn.Sequential()
        if self.input_dim != self.output_dim:
            shortcut_conv = nn.Conv1d(self.input_dim, self.output_dim, kernel_size=1, stride=1, padding=0, bias=False)
            self.shortcut.add_module('shortcut_conv', shortcut_conv)
            self.shortcut.add_module('shortcut_bn', nn.BatchNorm1d(self.output_dim))

        # Create individual attention branches
        self.attentions = nn.ModuleList([
            MultiHeadAttention(input_dim=self.input_dim,
                               attention_feature_dim=feat_dim,
                               output_dim=self.output_dim,
                               grid_size=grid_size,
                               grid_dim=grid_dim,
                               num_heads=num_heads,
                               use_scales=use_scales,
                               use_checkpoint=use_checkpoint)
            for feat_dim, grid_size, grid_dim, num_heads in zip(self.features_dims,
                                                                self.grid_sizes,
                                                                self.grid_dims,
                                                                self.num_heads_list)
        ])
        #self._reset_parameters()

    def _reset_parameters(self):
        xavier_uniform_(self.after[0].weight)
        if hasattr(self.shortcut, 'shortcut_conv'):
            xavier_uniform_(self.shortcut.shortcut_conv.weight)
    def forward(self, x, orig_points):
        """
        Forward pass for union attention.

        Args:
            x (Tensor): Input feature map of shape (B, C, P).
            orig_points (Tensor or tuple): Original point cloud or (orig_points, pts_padding).

        Returns:
            final_output (Tensor): Fused output after combining all attention branches.
            stats_all (list): List of statistics from each branch.
        """
        stats_all = []
        residual = self.shortcut(x) if len(self.shortcut) > 0 else x

        branch_outputs = []
        for attention in self.attentions:
            head_output, stats = attention(x, orig_points)
            branch_outputs.append(head_output)
            stats_all.append(stats)

        concatenated = torch.cat(branch_outputs, dim=1)
        gathered = self.after(concatenated)

        return residual + gathered, stats_all
