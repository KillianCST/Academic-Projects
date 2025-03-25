import os
import numpy as np
import torch
from torch.utils.data import Dataset
from data_processing.ply import read_ply
from data_processing.data_augmentation import RandomRotation_z, RandomNoise, center_data, normalize_data

class ModelNetDataset(Dataset):
    def __init__(self, root_dir, folder="train", rotate=False, noise=False):
        self.rotate = rotate
        self.noise = noise
        self.rotator = RandomRotation_z() if rotate else None
        self.noiser  = RandomNoise() if noise else None

        categories = sorted(d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d)))
        self.class_to_idx = {cat: i for i, cat in enumerate(categories)}
        self.samples = []
        for cat in categories:
            path = os.path.join(root_dir, cat, folder)
            for fname in os.listdir(path):
                if fname.endswith(".ply"):
                    pc = read_ply(os.path.join(path, fname))
                    pts = np.vstack((pc["x"], pc["y"], pc["z"])).T.astype(np.float32)
                    pts = center_data(np.expand_dims(pts, axis=0))[0]
                    pts = normalize_data(np.expand_dims(pts, axis=0))[0]
                    self.samples.append((pts, self.class_to_idx[cat]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        pointcloud, label = self.samples[idx]
        if self.rotate:
            pointcloud = self.rotator(pointcloud)
        if self.noise:
            pointcloud = self.noiser(pointcloud)
        pointcloud = torch.from_numpy(pointcloud).float()
        return {"pointcloud": pointcloud, "category": label}
