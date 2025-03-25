import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from data_processing.data_augmentation import RandomRotation_y, RandomNoise, center_data, normalize_data

class ScanObjectNNDataset(Dataset):
    def __init__(self, h5_path, rotate=False, noise=False, center=True, normalize=True):
        with h5py.File(h5_path, 'r') as f:
            self.data   = f['data'][:].astype(np.float32)
            self.labels = f['label'][:].astype(np.int64)
            # Convert mask: set -1 to 0 and leave other values as 1.
            mask_data   = f['mask'][:]
            binary_mask = np.ones(mask_data.shape, dtype=np.float32)
            binary_mask[mask_data == -1] = 0.0
            self.mask = binary_mask

        if center:
            self.data = center_data(self.data)
        if normalize:
            self.data = normalize_data(self.data)

        self.rotate = rotate
        self.noise  = noise
        self.rotator = RandomRotation_y() if rotate else None
        self.noiser  = RandomNoise()       if noise  else None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        pc    = self.data[idx]
        label = int(self.labels[idx])
        mask  = self.mask[idx]

        if self.rotate:
            pc = self.rotator(pc)
        if self.noise:
            pc = self.noiser(pc)

        return {
            'pointcloud': torch.from_numpy(pc).float(),  # (2048, 3)
            'category'  : label,
            'mask'      : torch.from_numpy(mask).float()    # (2048,)
        }
