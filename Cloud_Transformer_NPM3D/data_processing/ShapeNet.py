import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
import open3d as o3d
import open3d.visualization.gui as gui

def sphere_noise(batch: int, num_pts: int, device: torch.device) -> torch.Tensor:
    """Generate random points uniformly on unit sphere."""
    theta = 2 * np.pi * torch.rand(batch, num_pts, device=device)
    phi = torch.acos(1 - 2 * torch.rand(batch, num_pts, device=device))
    x = torch.sin(phi) * torch.cos(theta)
    y = torch.sin(phi) * torch.sin(theta)
    z = torch.cos(phi)
    return torch.stack([x, y, z], dim=1)


def resample_pcd(pcd: np.ndarray, n: int) -> np.ndarray:
    """Drop or duplicate points so that pcd has exactly n points."""
    idx = np.random.permutation(pcd.shape[0])
    if pcd.shape[0] < n:
        idx = np.concatenate([idx, np.random.randint(pcd.shape[0], size=n - pcd.shape[0])])
    return pcd[idx[:n]]


def partial_postprocess(partial_pcd: torch.Tensor, gt_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build clean encoder input and noisy generator input with mask.

    """
    batch_size, input_n, _ = partial_pcd.shape
    clean_list, noisy_list = [], []
    device = partial_pcd.device
    for b in range(batch_size):
        pts = partial_pcd[b]
        mask = ~(pts == 0.0).all(dim=1)
        valid_pts = pts[mask]
        # pad with uniform sphere noise to reach gt_size
        num_noise = gt_size - valid_pts.shape[0]
        noise = sphere_noise(1, num_noise, device)[0].permute(1, 0)
        valid_flag = torch.ones(valid_pts.shape[0], 1, device=device)
        noise_flag = torch.zeros(noise.shape[0], 1, device=device)
        part_labeled = torch.cat([valid_pts, valid_flag], dim=1)
        noise_labeled = torch.cat([noise, noise_flag], dim=1)
        noisy = torch.cat([noise_labeled, part_labeled], dim=0)
        # resample valid points to input_n
        clean_np = resample_pcd(valid_pts.cpu().numpy(), input_n)
        clean = torch.from_numpy(clean_np).to(device)
        clean_list.append(clean)
        noisy_list.append(noisy)
    return torch.stack(clean_list, dim=0), torch.stack(noisy_list, dim=0)


class ShapeNetCompletionDataset(Dataset):
    def __init__(self, root_dir, split='train', n_input=2048, n_output=16384):
        self.root_dir = root_dir
        self.split = split
        self.n_input = n_input
        self.n_output = n_output

        base_partial = os.path.join(root_dir, split, 'partial')
        base_complete = os.path.join(root_dir, split, 'complete')
        categories = sorted(os.listdir(base_partial))
        self.class_to_idx = {cat: i for i, cat in enumerate(categories)}
        self.samples = []

        for cat in categories:
            complete_dir = os.path.join(base_complete, cat)
            partial_cat = os.path.join(base_partial, cat)
            for object_id in os.listdir(partial_cat):
                complete_file = os.path.join(complete_dir, f"{object_id}.pcd")
                if not os.path.isfile(complete_file):
                    continue
                view_dir = os.path.join(partial_cat, object_id)
                for view_file in os.listdir(view_dir):
                    if view_file.endswith('.pcd'):
                        self.samples.append({
                            'partial_path': os.path.join(view_dir, view_file),
                            'complete_path': complete_file,
                            'label': self.class_to_idx[cat]
                        })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        entry = self.samples[idx]
        partial_np = np.asarray(o3d.io.read_point_cloud(entry['partial_path']).points, dtype=np.float32)
        complete_np = np.asarray(o3d.io.read_point_cloud(entry['complete_path']).points, dtype=np.float32)

        # Train‑only mirror augmentation
        if self.split == 'train' and random.random() < 0.5:
            partial_np[:, 0] *= -1
            complete_np[:, 0] *= -1

        # Fixed‑size sampling
        perm = np.random.choice(partial_np.shape[0], self.n_input, replace=partial_np.shape[0] < self.n_input)
        partial = torch.from_numpy(partial_np[perm]).float().unsqueeze(0)

        perm_c = np.random.choice(complete_np.shape[0], self.n_output, replace=complete_np.shape[0] < self.n_output)
        complete = torch.from_numpy(complete_np[perm_c]).float().unsqueeze(0)

        p_enc, p_noise = partial_postprocess(partial, self.n_output)

         # Squeeze batched dims
        complete = complete.squeeze(0)         # (n_output, 3)
        partial_enc = p_enc.squeeze(0)         # (n_input, 3)
        partial_noise = p_noise.squeeze(0)     # (n_output, 4)


        return {
            'partial_enc': partial_enc,
            'partial_noise': partial_noise,
            'complete_cloud': complete,
            'category': entry['label']
        }



def visualize_completion(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    batch_idx: int = 0,
    num_samples: int = 1
):
    model.eval()
    it = iter(dataloader)
    for _ in range(batch_idx + 1):
        batch = next(it)

    partial_enc = batch['partial_enc'].to(device)
    partial_noise = batch['partial_noise'].to(device)

    with torch.no_grad():
      
        pred = model(partial_noise, partial_enc).permute(0, 2, 1)  # → [B, N, 3]

    num_samples = min(num_samples, pred.shape[0])
    try:
        gui.Application.instance.initialize()
    except RuntimeError:
        pass

    def make_pcd(points, color):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.paint_uniform_color(color)
        return pcd

    import numpy as np

    for i in range(num_samples):
        partial_np = partial_enc[i].cpu().numpy().astype(np.float64)
        pred_np    = pred[i].cpu().numpy()

        partial_np = partial_np.reshape(-1, 3).astype(np.float64)
        pred_np    = pred_np.reshape(-1, 3).astype(np.float64)

        pcd_left  = make_pcd(partial_np, [0.7, 0.7, 0.7])
        pcd_right = make_pcd(pred_np,    [1.0, 0.0, 0.0])

        pcd_left.translate((-0.5, 0, 0))
        pcd_right.translate(( 0.5, 0, 0))
        o3d.visualization.draw_geometries([pcd_left, pcd_right])




