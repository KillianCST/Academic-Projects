import torch
from torch import nn

from torch.nn.init import xavier_uniform_, constant_
from cloud_transformer.ResBlock.v2v_groups import Res3DBlock, Pool3DBlock
from cloud_transformer.ResBlock.unet_parts import Res2DBlock
from cloud_transformer.multihead_union import MultiHeadUnionAttention
from cloud_transformer.multihead_pool import MultiHeadPool
from cloud_transformer.multihead_adain import MultiHeadUnionAdaIn, _apply_style
from cloud_transformer.utils_ct import AdaIn1dUpd

class CT_Encoder(nn.Module):
    def __init__(self, model_dim=512, heads=16, num_layers=4, use_scales=True, use_checkpoint=False):
        super().__init__()
        self.model_dim = model_dim
        self.heads = heads
        self.num_layers = num_layers
        self.use_scales = use_scales
        self.use_checkpoint = use_checkpoint

        self.first_process = nn.Sequential(
            nn.Conv1d(3, model_dim, kernel_size=1, bias=False),
            nn.BatchNorm1d(model_dim),
            nn.ReLU(inplace=True)
        )

        self.attentions_encoder = nn.ModuleList(
            [branch for _ in range(self.num_layers) for branch in [
                MultiHeadUnionAttention(
                    input_dim=self.model_dim,
                    features_dims=[4, 4],
                    num_heads_list=[self.heads, self.heads],
                    grid_sizes=[64, 16],
                    grid_dims=[2, 3],
                    output_dim=self.model_dim,
                    use_scales=self.use_scales,
                    use_checkpoint=self.use_checkpoint
                ),
                MultiHeadUnionAttention(
                    input_dim=self.model_dim,
                    features_dims=[4 * 4, 4 * 4],
                    num_heads_list=[self.heads, self.heads],
                    grid_sizes=[32, 8],
                    grid_dims=[2, 3],
                    output_dim=self.model_dim,
                    use_scales=self.use_scales,
                    use_checkpoint=self.use_checkpoint
                ),
                MultiHeadUnionAttention(
                    input_dim=self.model_dim,
                    features_dims=[4 * 4, 4 * 8],
                    num_heads_list=[self.heads, self.heads],
                    grid_sizes=[8, 4],
                    grid_dims=[2, 3],
                    output_dim=self.model_dim,
                    use_scales=self.use_scales,
                    use_checkpoint=self.use_checkpoint
                )
            ]]
        )

        self.pool3d = MultiHeadPool(
            model_dim=self.model_dim,
            feature_dim=4 * 8,
            grid_size=8,
            grid_dim=3,
            num_heads=heads,
            scales=self.use_scales,
            use_checkpoint=self.use_checkpoint
        )
        pool3d_out_channels = 32 * heads
        self.after_pool3d = nn.Sequential(
            Res3DBlock(pool3d_out_channels, 64 * heads, groups=16),
            Pool3DBlock(2),
            Res3DBlock(64 * heads, 64 * heads, groups=16),
            Pool3DBlock(2),
            Res3DBlock(64 * heads, 64 * heads, groups=16),
            nn.AdaptiveAvgPool3d((1, 1, 1))
        )

        self.pool2d = MultiHeadPool(
            model_dim=self.model_dim,
            feature_dim=4 * 4,
            grid_size=16,
            grid_dim=2,
            num_heads=heads,
            scales=self.use_scales,
            use_checkpoint=self.use_checkpoint
        )
        pool2d_out_channels = 16 * heads
        self.after_pool2d = nn.Sequential(
            Res2DBlock(pool2d_out_channels, 32 * heads, groups=16),
            nn.MaxPool2d(2),
            Res2DBlock(32 * heads, 64 * heads, groups=16),
            nn.MaxPool2d(2),
            Res2DBlock(64 * heads, 64 * heads, groups=16),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.class_vector = nn.Sequential(
            nn.Linear(64 * heads + 64 * heads, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True)
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

        B = x.size(0)
        x3d, _ = self.pool3d(x, orig)
        x3d = self.after_pool3d(x3d).view(B, -1)

        x2d, _ = self.pool2d(x, orig)
        x2d = self.after_pool2d(x2d).view(B, -1)

        latent_feature = self.class_vector(torch.cat([x2d, x3d], dim=-1))
        return latent_feature


class CT_Completion(nn.Module):
    def __init__(self, num_latent=512, model_dim=512, heads=16, num_layers=4, use_scales=True, use_checkpoint=False):
        super().__init__()
        self.num_latent = num_latent
        self.model_dim = model_dim
        self.heads = heads
        self.num_layers = num_layers
        self.use_scales = use_scales
        self.use_checkpoint = use_checkpoint

        # Encoder branch (input: (B, P, 3))
        self.encoder = CT_Encoder(
            model_dim=model_dim,
            heads=heads,
            num_layers=num_layers,
            use_scales=use_scales,
            use_checkpoint=use_checkpoint
        )

        # Map encoder features (1024–dim) to a latent style vector
        self.mapping = nn.Sequential(
            nn.Linear(1024, num_latent),
            nn.ReLU(inplace=True)
        )

        # Decoder “start” block: input is expected to have 4 channels
        self.start = nn.Sequential(
            nn.Conv1d(in_channels=4, out_channels=model_dim, kernel_size=1, bias=False),
            AdaIn1dUpd(self.model_dim, num_latent),
            nn.ReLU(inplace=True)
        )

        # Build decoder attention layers using MultiHeadAdaIn blocks
        self.num_decoder_layers = num_layers
        self.attentions_decoder = nn.ModuleList(
            [branch for _ in range(self.num_decoder_layers) for branch in [
                MultiHeadUnionAdaIn(
                    input_dim=self.model_dim,
                    feature_dims=[4, 4],
                    num_heads_list=[self.heads, self.heads],
                    grid_sizes=[64, 16],
                    grid_dims=[2, 3],
                    output_dim=self.model_dim,
                    n_latent=self.num_latent,
                    use_scales=self.use_scales,
                    use_checkpoint=self.use_checkpoint
                ),
                MultiHeadUnionAdaIn(
                    input_dim=self.model_dim,
                    feature_dims=[4*4, 4*4],
                    num_heads_list=[self.heads, self.heads],
                    grid_sizes=[32, 8],
                    grid_dims=[2, 3],
                    output_dim=self.model_dim,
                    n_latent=self.num_latent,
                    use_scales=self.use_scales,
                    use_checkpoint=self.use_checkpoint
                ),
                MultiHeadUnionAdaIn(
                    input_dim=self.model_dim,
                    feature_dims=[4*4, 4*8],
                    num_heads_list=[self.heads, self.heads],
                    grid_sizes=[8, 4],
                    grid_dims=[2, 3],
                    output_dim=self.model_dim,
                    n_latent=self.num_latent,
                    use_scales=self.use_scales,
                    use_checkpoint=self.use_checkpoint
                )
            ]]
        )

        # Final reconstruction block
        self.final = nn.Sequential(
            nn.Conv1d(in_channels=model_dim + 4, out_channels=model_dim, kernel_size=1, bias=False),
            AdaIn1dUpd(self.model_dim, num_latent),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=model_dim, out_channels=3, kernel_size=1)
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

    def forward(self, noise, input):
        # Encoder branch: input shape (B, P, 3)
        latent_feature = self.encoder(input)
        
        B = latent_feature.size(0)
        latent_feature = latent_feature.view(B, -1)  # (B, 1024)
        latent_style = self.mapping(latent_feature)    # (B, num_latent)

        # Convert noise to channels‑first for all Conv1d/AdaIN layers
        noise_feat = noise.permute(0, 2, 1)            # → (B, 4, P)
        x_dec = _apply_style(self.start, noise_feat, latent_style)

        orig_points = noise_feat[:, :3, :]

        # Decoder
        for decoder_attention in self.attentions_decoder:
            x_dec, _ = decoder_attention(x_dec, orig_points, latent_style)
            
       
        # Final reconstruction
        x_cat = torch.cat([x_dec, noise_feat], dim=1)  # (B, model_dim+4, P)
        
        x_out = _apply_style(self.final, x_cat, latent_style)

        return x_out
