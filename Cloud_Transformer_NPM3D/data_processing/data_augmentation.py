import numpy as np
import random
import math

class RandomRotation_z(object):
    def __call__(self, pointcloud):
        theta = random.random() * 2. * math.pi
        rot_matrix = np.array([[ math.cos(theta), -math.sin(theta),      0],
                               [ math.sin(theta),  math.cos(theta),      0],
                               [0,                               0,      1]])
        rot_pointcloud = rot_matrix.dot(pointcloud.T).T
        return rot_pointcloud
    
class RandomRotation_y(object):
    def __call__(self, pointcloud):
        theta = random.random() * 2. * math.pi
        rot_matrix = np.array([[ math.cos(theta), 0, math.sin(theta)],
                               [ 0,               1, 0             ],
                               [-math.sin(theta), 0, math.cos(theta)]])
        rot_pointcloud = rot_matrix.dot(pointcloud.T).T
        return rot_pointcloud

class RandomNoise(object):
    def __init__(self, sigma=0.01, clip=0.05):
        assert clip > 0
        self.sigma = sigma
        self.clip = clip

    def __call__(self, pointcloud):
        noise = np.random.normal(0, self.sigma, pointcloud.shape)
        noise = np.clip(noise, -self.clip, self.clip)
        noisy = pointcloud + noise

        # Project any point whose norm > 1 back to the unit sphere
        norms = np.linalg.norm(noisy, axis=1)
        mask = norms > 1.0
        noisy[mask] = noisy[mask] / norms[mask][:, None]

        return noisy


def center_data(pcs):
    for i in range(pcs.shape[0]):
        centroid = np.mean(pcs[i], axis=0)
        pcs[i] = pcs[i] - centroid
    return pcs

def normalize_data(pcs):
    for i in range(pcs.shape[0]):
        d = np.max(np.linalg.norm(pcs[i], axis=1))
        pcs[i] = pcs[i] / d
    return pcs