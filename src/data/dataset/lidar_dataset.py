import os
import numpy as np
import torch
from torch.utils.data import Dataset

class MultiBagE2EDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.samples = []
        self.transform = transform
        root_dir = os.path.expanduser(root_dir)

        for bag_name in os.listdir(root_dir):
            bag_dir = os.path.join(root_dir, bag_name)
            if not os.path.isdir(bag_dir):
                continue

            scans_path = os.path.join(bag_dir, 'scans.npy')
            steers_path = os.path.join(bag_dir, 'steers.npy')
            speeds_path = os.path.join(bag_dir, 'speeds.npy')

            if not (os.path.exists(scans_path) and os.path.exists(steers_path) and os.path.exists(speeds_path)):
                print(f"[WARN] Skipping incomplete bag: {bag_name}")
                continue

            scans = np.load(scans_path)
            steers = np.load(steers_path)
            speeds = np.load(speeds_path)

            for i in range(len(scans)):
                self.samples.append((scans[i], steers[i], speeds[i]))

        print(f"[INFO] Loaded {len(self.samples)} samples from {root_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        scan, steer, speed = self.samples[idx]

        if self.transform:
            scan, steer, speed = self.transform(scan, steer, speed)

        return {
            'scan': torch.from_numpy(scan.astype(np.float32)),
            'steer': torch.tensor(steer, dtype=torch.float32),
            'speed': torch.tensor(speed, dtype=torch.float32)
        }

