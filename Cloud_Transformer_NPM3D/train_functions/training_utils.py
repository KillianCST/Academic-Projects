from itertools import islice
import open3d as o3d
import torch
import numpy as np

def _slice_loader(loader, subsample):
    if subsample is None:
        return loader
    total = len(loader)
    if isinstance(subsample, float):
        count = max(1, int(subsample * total))
    else:
        count = min(subsample, total)
    return islice(loader, count)

def _count_batches(loader, subsample):
    total = len(loader)
    if subsample is None:
        return total
    if isinstance(subsample, float):
        return max(1, int(subsample * total))
    return min(subsample, total)



def compute_batch_f1(pred: torch.Tensor, gt: torch.Tensor, threshold: float = 0.01) -> float:
    """
    Compute mean F1@threshold between pred and gt point clouds.
    pred, gt: tensors of shape (B, N, 3)
    threshold: distance cutoff
    """
    batch_size = pred.shape[0]
    f1_scores = []

    for i in range(batch_size):
        pc_pred = o3d.geometry.PointCloud()
        pc_pred.points = o3d.utility.Vector3dVector(pred[i].cpu().numpy())

        pc_gt = o3d.geometry.PointCloud()
        pc_gt.points = o3d.utility.Vector3dVector(gt[i].cpu().numpy())

        # Distance from each pred point to nearest gt
        d_pred = np.asarray(pc_pred.compute_point_cloud_distance(pc_gt))
        # Distance from each gt point to nearest pred
        d_gt   = np.asarray(pc_gt.compute_point_cloud_distance(pc_pred))

        precision = (d_pred <= threshold).sum() / max(len(d_pred), 1)
        recall    = (d_gt   <= threshold).sum() / max(len(d_gt),   1)

        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        f1_scores.append(f1)

    return float(np.mean(f1_scores))

