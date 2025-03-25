import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_
from torch.utils.checkpoint import checkpoint
from cloud_transformer.core import Splat, DifferentiablePositions
from cloud_transformer.utils_ct import PlaneTransformer, VolTransformer

class MultiHeadPool(nn.Module):
    def __init__(self,
                 model_dim,
                 feature_dim,
                 grid_size,
                 grid_dim,
                 num_heads,
                 scales=False,
                 use_checkpoint=False):
        """
        Args:
            model_dim (int): Input channel dimension.
            feature_dim (int): Feature dimension per head.
            grid_size (int): Size of the grid/tensor.
            grid_dim (int): Dimensionality of the grid (2 or 3).
            num_heads (int): Number of attention heads.
            scales (bool): Whether to use scale transformation.
            use_checkpoint (bool): If True, use checkpointing in the forward pass.
        """
        super().__init__()
        assert grid_dim in (2, 3), "grid_dim must be either 2 or 3."

        self.feature_dim = feature_dim
        self.model_dim = model_dim
        self.grid_size = grid_size
        self.grid_dim = grid_dim
        self.num_heads = num_heads
        self.use_checkpoint = use_checkpoint

        # Predict key offsets (first num_heads*3 channels) and attention values (remaining channels)
        self.keys_values_pred = nn.Sequential(
            nn.Conv1d(self.model_dim,
                      self.num_heads * (self.feature_dim + 3),
                      kernel_size=1,
                      bias=False)
        )

        # Batch normalization for keys and values separately
        self.values_bn = nn.BatchNorm1d(self.num_heads * self.feature_dim)
        self.key_bn = nn.BatchNorm1d(self.num_heads * 3)
        # Initialize key batch norm weights to 0 as intended
        nn.init.zeros_(self.key_bn.weight)

        # Create differentiable positions module to compute lattice coordinates
        self.diff_pos_generator = DifferentiablePositions(tensor_size=self.grid_size,
                                                            dim=self.grid_dim,
                                                            heads=self.num_heads)

        # Splatting layer to project features onto the lattice
        self.splat = Splat(tensor_size=self.grid_size,
                           dim=self.grid_dim,
                           heads=self.num_heads)

        # Choose transformation module based on grid dimensionality
        if self.grid_dim == 3:
            self.transform = VolTransformer(self.num_heads, scales=scales)
        else:
            self.transform = PlaneTransformer(self.num_heads, scales=scales)

        self._reset_parameters()
    def _reset_parameters(self):
        #xavier_uniform_(self.keys_values_pred[0].weight)
        torch.nn.init.zeros_(self.key_bn.weight)

    def _forward_body(self, input_tensor, original_points):
        """
        Performs the heavy computation of the forward pass.
        
        Args:
            input_tensor (Tensor): Input features of shape (B, model_dim, P).
            original_points (Tensor): Original point cloud coordinates of shape (B, 3, P).
        
        Returns:
            keys (Tensor): Transformed keys with shape (B, num_heads * grid_dim, P).
            lattice (Tensor): Lattice coordinates (after tanh) of shape (B, num_heads * grid_dim, P).
            local_coord (Tensor): Local coordinates from the differentiable positions module.
            flattened_index (Tensor): Flattened index corresponding to the lattice.
            splat_out (Tensor): Output of the splatting operation.
        """
        # Predict keys offset and values
        key_values = self.keys_values_pred(input_tensor)  # Shape: (B, num_heads*(feature_dim+3), P)
        
        # Extract keys offset and values using separate batch norms
        keys_offset = self.key_bn(key_values[:, :self.num_heads * 3])
        values = self.values_bn(key_values[:, self.num_heads * 3:])
        
        # Reshape keys_offset to (B, num_heads, 3, P) and add to original point cloud
        keys_offset = keys_offset.view(input_tensor.shape[0], self.num_heads, 3, input_tensor.shape[-1])
        transformed_keys = self.transform(original_points[:, None] + keys_offset)
        
        # Flatten transformed keys to (B, num_heads * grid_dim, P)
        keys = transformed_keys.reshape(input_tensor.shape[0], self.num_heads * self.grid_dim, input_tensor.shape[-1])
        lattice = torch.tanh(keys)
        
        # Compute lattice local coordinates and indices
        local_coord, flattened_index = self.diff_pos_generator(lattice)
        
        # Project the attention values onto the lattice
        splat_out = self.splat(local_coord, flattened_index, values)
        return keys, lattice, splat_out

    def forward(self, input_tensor, original_points, return_lattice=False):
        """
        Args:
            input_tensor (Tensor): Input feature map of shape (B, model_dim, P).
            original_points (Tensor): Original point cloud (B, 3, P).
            return_lattice (bool): If True, also return the computed lattice.
        
        Returns:
            result (Tensor or tuple): The splatted features, or (features, lattice) if return_lattice is True.
            stats (tuple): A tuple of statistics (occupancy, mean(keys), var(keys), None).
        """
        # Use checkpointing if enabled and in training mode
        if self.use_checkpoint and self.training:
            keys, lattice, splat_out = checkpoint(
                self._forward_body, input_tensor, original_points, use_reentrant=False)
        else:
            keys, lattice, splat_out = self._forward_body(
                input_tensor, original_points)

        # Compute occupancy statistics
        with torch.no_grad():
            occupancy = (torch.abs(splat_out) > 1e-9).sum().float() / (keys.size(0) * self.feature_dim * self.num_heads)
            keys_mean = torch.mean(keys).detach()
            keys_var = torch.var(keys).detach()
            stats = (occupancy, keys_mean, keys_var, None)

        result = splat_out
        if return_lattice:
            result = (result, lattice)
        return result, stats
