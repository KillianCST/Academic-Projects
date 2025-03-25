import torch
from torch import nn
from torch.nn.init import xavier_uniform_, constant_

from cloud_transformer.ResBlock.v2v_groups import Res3DBlock, Pool3DBlock
from cloud_transformer.ResBlock.unet_parts import Res2DBlock
from cloud_transformer.multihead_union import MultiHeadUnionAttention
from cloud_transformer.multihead_pool import MultiHeadPool


class CT_Classifier(nn.Module):
    def __init__(
        self,
        n_classes: int = 15,
        model_dim: int = 512,
        heads: int = 16,
        num_layers: int = 4,
        dropout: float = 0.5,
        use_scales: bool = True,
        use_checkpoint: bool = False,
    ):
        super().__init__()

        self.n_classes = n_classes
        self.model_dim = model_dim
        self.heads = heads
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_checkpoint = use_checkpoint
        self.use_scales = use_scales

        self.first_process = nn.Sequential(
            nn.Conv1d(3, model_dim, kernel_size=1, bias=False),
            nn.BatchNorm1d(model_dim),
            nn.ReLU(inplace=True),
        )

        self.attentions_encoder = nn.ModuleList(
            [
                branch
                for _ in range(num_layers)
                for branch in [
                    MultiHeadUnionAttention(
                        input_dim=model_dim,
                        features_dims=[4, 4],
                        num_heads_list=[heads, heads],
                        grid_sizes=[128, 32],
                        grid_dims=[2, 3],
                        output_dim=model_dim,
                        use_scales=use_scales,
                        use_checkpoint=use_checkpoint,
                    ),
                    MultiHeadUnionAttention(
                        input_dim=model_dim,
                        features_dims=[16, 16],
                        num_heads_list=[heads, heads],
                        grid_sizes=[64, 16],
                        grid_dims=[2, 3],
                        output_dim=model_dim,
                        use_scales=use_scales,
                        use_checkpoint=use_checkpoint,
                    ),
                    MultiHeadUnionAttention(
                        input_dim=model_dim,
                        features_dims=[16, 32],
                        num_heads_list=[heads, heads],
                        grid_sizes=[16, 8],
                        grid_dims=[2, 3],
                        output_dim=model_dim,
                        use_scales=use_scales,
                        use_checkpoint=use_checkpoint,
                    ),
                ]
            ]
        )

        self.pool3d = MultiHeadPool(
            model_dim=model_dim,
            feature_dim=32,
            grid_size=8,
            grid_dim=3,
            num_heads=heads,
            scales=use_scales,
            use_checkpoint=use_checkpoint,
        )
        pool3d_out = 32 * heads
        self.after_pool3d = nn.Sequential(
            Res3DBlock(pool3d_out, 64 * heads, groups=16),
            Pool3DBlock(2),
            Res3DBlock(64 * heads, 64 * heads, groups=16),
            Pool3DBlock(2),
            Res3DBlock(64 * heads, 64 * heads, groups=16),
            nn.AdaptiveAvgPool3d((1, 1, 1)),
        )

        self.pool2d = MultiHeadPool(
            model_dim=model_dim,
            feature_dim=16,
            grid_size=16,
            grid_dim=2,
            num_heads=heads,
            scales=use_scales,
            use_checkpoint=use_checkpoint,
        )
        pool2d_out = 16 * heads
        self.after_pool2d = nn.Sequential(
            Res2DBlock(pool2d_out, 32 * heads, groups=16),
            nn.MaxPool2d(2),
            Res2DBlock(32 * heads, 64 * heads, groups=16),
            nn.MaxPool2d(2),
            Res2DBlock(64 * heads, 64 * heads, groups=16),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.class_vector = nn.Sequential(
            nn.Linear(64 * heads * 2, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
        )
        self.class_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(1024, n_classes),
        )

        self.mask_head = nn.Sequential(
            nn.Conv1d(model_dim + 1024, 256, kernel_size=1, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(256, 1, kernel_size=1),
        )

      
    def _reset_parameters(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)):
                xavier_uniform_(m.weight)
                if m.bias is not None:
                    constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                constant_(m.weight, 1)
                constant_(m.bias, 0)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        orig = x

        x = self.first_process(x)
        for attention in self.attentions_encoder:
            x, _ = attention(x, orig)

        to_3d, _ = self.pool3d(x, orig)
        pooled_3d = self.after_pool3d(to_3d).reshape(x.size(0), -1)

        to_2d, _ = self.pool2d(x, orig)
        pooled_2d = self.after_pool2d(to_2d).reshape(x.size(0), -1)

        class_vect = self.class_vector(torch.cat([pooled_2d, pooled_3d], dim=1))
        class_pred = self.class_head(class_vect)

        mask_pred = (
            self.mask_head(torch.cat([x, class_vect.unsqueeze(-1).expand(-1, -1, x.size(-1))], dim=1))
            .unsqueeze(2)
        )

        return class_pred, mask_pred
