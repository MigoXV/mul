from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from fairseq.dataclass import FairseqDataclass
from fairseq.tasks import FairseqTask, register_task
from omegaconf import MISSING

from mul.datasets.celeba import CelebADataset
from torchvision import transforms


@dataclass
class BinClassConfig(FairseqDataclass):
    train_h5_path: Optional[str] = None
    valid_h5_path: Optional[str] = None


@register_task("bin-class", dataclass=BinClassConfig)
class BinClassTask(FairseqTask):
    def __init__(self, config: BinClassConfig):
        super().__init__(config)
        self.config = config
        self.train_h5_path = Path(config.train_h5_path)
        self.valid_h5_path = Path(config.valid_h5_path)
        self.train_transform = transforms.Compose(
            [
                transforms.Resize(128),
                transforms.RandomHorizontalFlip(),
                transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
            ]
        )

        self.test_transform = transforms.Compose(
            [transforms.Resize(128), transforms.ToTensor()]
        )

    def load_dataset(self, split, **kwargs):
        if split == "train":
            self.datasets[split] = CelebADataset(
                self.train_h5_path, transform=self.train_transform
            )
        elif split == "valid":
            self.datasets[split] = CelebADataset(
                self.valid_h5_path, transform=self.test_transform
            )
        else:
            raise KeyError(f"Invalid split: {split}")

    @property
    def target_dictionary(self):
        """Return the target :class:`~fairseq.data.Dictionary` (if applicable
        for this task)."""
        return None
