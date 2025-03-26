import torch
import numpy as np
import torch.nn as nn
from pytorch3d.transforms.so3 import so3_exponential_map


def trilinear_coords(keys):
    assert (keys.shape[1] == 3)

    spread = torch.from_numpy(np.array([[0, 0, 0],
                                        [1, 0, 0],
                                        [0, 1, 0],
                                        [1, 1, 0],
                                        [0, 0, 1],
                                        [1, 0, 1],
                                        [0, 1, 1],
                                        [1, 1, 1]])).to(keys.device)

    floored = keys.floor()
    ix, iy, iz = keys[:, 0], keys[:, 1], keys[:, 2]
    ix_tnw, iy_tnw, iz_tnw = floored[:, 0], floored[:, 1], floored[:, 2]

    ix_tne = ix_tnw + 1
    iy_tne = iy_tnw
    iz_tne = iz_tnw

    ix_tsw = ix_tnw
    iy_tsw = iy_tnw + 1
    iz_tsw = iz_tnw

    ix_tse = ix_tnw + 1
    iy_tse = iy_tnw + 1
    iz_tse = iz_tnw

    ix_bnw = ix_tnw
    iy_bnw = iy_tnw
    iz_bnw = iz_tnw + 1

    ix_bne = ix_tnw + 1
    iy_bne = iy_tnw
    iz_bne = iz_tnw + 1

    ix_bsw = ix_tnw
    iy_bsw = iy_tnw + 1
    iz_bsw = iz_tnw + 1

    ix_bse = ix_tnw + 1
    iy_bse = iy_tnw + 1
    iz_bse = iz_tnw + 1

    tnw = (ix_bse - ix) * (iy_bse - iy) * (iz_bse - iz)
    tne = (ix - ix_bsw) * (iy_bsw - iy) * (iz_bsw - iz)
    tsw = (ix_bne - ix) * (iy - iy_bne) * (iz_bne - iz)
    tse = (ix - ix_bnw) * (iy - iy_bnw) * (iz_bnw - iz)
    bnw = (ix_tse - ix) * (iy_tse - iy) * (iz - iz_tse)
    bne = (ix - ix_tsw) * (iy_tsw - iy) * (iz - iz_tsw)
    bsw = (ix_tne - ix) * (iy - iy_tne) * (iz - iz_tne)
    bse = (ix - ix_tnw) * (iy - iy_tnw) * (iz - iz_tnw)

    coordinates = torch.stack([tnw, tne, tsw, tse, bnw, bne, bsw, bse], dim=1)

    return coordinates, floored[:, None].long() + spread[:, :, None]


def bilinear_coords(keys):
    assert (keys.shape[1] == 2)

    spread = torch.from_numpy(np.array([[0, 0],
                                        [1, 0],
                                        [0, 1],
                                        [1, 1]])).to(keys.device)

    floored = keys.floor()
    ix, iy = keys[:, 0], keys[:, 1]
    ix_nw, iy_nw = floored[:, 0], floored[:, 1]

    ix_ne = ix_nw + 1
    iy_ne = iy_nw

    ix_sw = ix_nw
    iy_sw = iy_nw + 1

    ix_se = ix_nw + 1
    iy_se = iy_nw + 1

    nw = (ix_se - ix) * (iy_se - iy)
    ne = (ix - ix_sw) * (iy_sw - iy)
    sw = (ix_ne - ix) * (iy - iy_ne)
    se = (ix - ix_nw) * (iy - iy_nw)

    coordinates = torch.stack([nw, ne, sw, se], dim=1)

    return coordinates, floored[:, None].long() + spread[:, :, None]



class VolTransformer(nn.Module):
    def __init__(self, heads, scales=False):
        super().__init__()
        self.heads = heads

        self.log_R = nn.Parameter(torch.randn(self.heads, 3, dtype=torch.float32),
                                  requires_grad=True)
        self.shift = nn.Parameter(torch.zeros(self.heads, 3, dtype=torch.float32),
                                  requires_grad=True)

        self.do_scales = scales

        if self.do_scales:
            self.scales = nn.Parameter(torch.ones(self.heads, 3, dtype=torch.float32),
                                       requires_grad=True)

    def forward(self, pcd):
        # pcd [b, h, c, p]
        pcd = pcd + self.shift[None, :, :, None]

        pcd = torch.einsum('bhcp,hcn->bhnp', [pcd, so3_exponential_map(self.log_R)])

        if self.do_scales:
            return pcd * self.scales[None, :, :, None]
        else:
            return pcd


class PlaneTransformer(nn.Module):
    def __init__(self, heads, scales=False):
        super().__init__()
        self.heads = heads

        self.log_R = nn.Parameter(torch.randn(self.heads, 3, dtype=torch.float32),
                                  requires_grad=True)
        self.shift = nn.Parameter(torch.zeros(self.heads, 3, dtype=torch.float32),
                                  requires_grad=True)

        self.do_scales = scales

        if self.do_scales:
            self.scales = nn.Parameter(torch.ones(self.heads, 2, dtype=torch.float32),
                                       requires_grad=True)

    def forward(self, pcd):
        # pcd [b, h, c, p]
        pcd = pcd + self.shift[None, :, :, None]
        pcd = torch.einsum('bhcp,hcn->bhnp', [pcd, so3_exponential_map(self.log_R)])

        if self.do_scales:
            return pcd[:, :, :2] * self.scales[None, :, :, None]
        else:
            return pcd[:, :, :2]
        

class AdaIn1dUpd(nn.Module):
    def __init__(self, num_features, num_latent):
        super().__init__()
        self.num_features = num_features
        self.num_latent = num_latent

        self.instance_norm = nn.InstanceNorm1d(self.num_features, eps=1e-5, affine=False)
        self.linear = nn.Linear(self.num_latent, self.num_features * 2)

    def forward(self, x, z):
        x = self.instance_norm(x)

        var_bias = self.linear(z).reshape(-1, 2, self.num_features)

        return x * (var_bias[:, 0][:, :, None] + 1) + var_bias[:, 1][:, :, None]