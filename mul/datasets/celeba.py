import torch
from PIL import Image
from pathlib import Path
import h5py
import os
from fairseq.data import FairseqDataset
import numpy as np


class CelebADataset(FairseqDataset):
    def __init__(self, h5_path: Path, transform=None):
        self.h5_path = h5_path
        self.transform = transform
        with h5py.File(h5_path, "r") as f:
            self.total_len = len(f["images"])

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        with h5py.File(self.h5_path, "r") as f:
            img = f["images"][idx]
            label = f["is_male"][idx]
        return img, label

    def collater(self, samples):
        images = np.stack([s[0] for s in samples]).astype(np.float32)
        images = torch.from_numpy(images)
        images = images.permute(0, 3, 1, 2)
        labels = np.array([s[1] for s in samples]).astype(np.float32)
        labels = torch.from_numpy(labels)
        return images, labels

    def num_tokens(self, index):
        return 1
