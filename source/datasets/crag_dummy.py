import logging

import lightning
import numpy as np
import torch
from torch.utils.data import Dataset

from .crag import CragDataset

logger = logging.getLogger("trimesh")
logger.setLevel(logging.ERROR)


class DummyCragDataset(Dataset):
    """Lightweight synthetic dataset to benchmark data loading overhead."""

    def __init__(
        self,
        *,
        length: int = 256,
        category: str = "dummy",
        min_parts: int = 2,
        max_parts: int = 6,
        num_points_per_part: int = 256,
        num_points_per_object: int = 2048,
        num_renderings: int = 0,
        seed: int | None = None,
        **kwargs,
    ):
        super().__init__()
        self.length = length
        self.category = category
        self.min_parts = min_parts
        self.max_parts = max_parts
        self.num_points_per_part = num_points_per_part
        self.num_points_per_object = num_points_per_object
        self.num_renderings = num_renderings
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        num_parts = int(self.rng.integers(self.min_parts, self.max_parts + 1))
        points_per_part = np.full(num_parts, self.num_points_per_part, dtype=np.int32)
        total_points = int(points_per_part.sum())

        pointclouds = self.rng.standard_normal((total_points, 3), dtype=np.float32)
        pointclouds_gt = pointclouds.copy()

        pointclouds_normals = self.rng.standard_normal((total_points, 3), dtype=np.float32)
        normal_norm = np.linalg.norm(pointclouds_normals, axis=1, keepdims=True)
        pointclouds_normals /= np.clip(normal_norm, a_min=1e-6, a_max=None)
        pointclouds_normals_gt = pointclouds_normals.copy()

        quaternions = np.tile(np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32), (num_parts, 1))
        translations = np.zeros((num_parts, 3), dtype=np.float32)
        ref_part = np.zeros((num_parts,), dtype=bool)
        ref_part[self.rng.integers(0, num_parts)] = True

        full_pointcloud = self.rng.standard_normal((self.num_points_per_object, 3), dtype=np.float32)
        full_pointcloud_normal = self.rng.standard_normal((self.num_points_per_object, 3), dtype=np.float32)
        full_pointcloud_normal /= np.clip(
            np.linalg.norm(full_pointcloud_normal, axis=1, keepdims=True),
            a_min=1e-6,
            a_max=None,
        )

        renderings = torch.zeros((self.num_renderings, 3, 224, 224), dtype=torch.float32)

        return {
            "name": f"{self.category}/dummy_{index}",
            "num_parts": num_parts,
            "pointclouds": torch.from_numpy(pointclouds),
            "pointclouds_gt": torch.from_numpy(pointclouds_gt),
            "pointclouds_normals": torch.from_numpy(pointclouds_normals),
            "pointclouds_normals_gt": torch.from_numpy(pointclouds_normals_gt),
            "quaternions": torch.from_numpy(quaternions),
            "translations": torch.from_numpy(translations),
            "points_per_part": torch.from_numpy(points_per_part),
            "points_per_object": torch.tensor(self.num_points_per_object, dtype=torch.int32),
            "ref_part": torch.from_numpy(ref_part),
            "init_rot": torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float32),
            "pieces": ",".join([f"piece_{i}" for i in range(num_parts)]),
            "full_pointcloud": torch.from_numpy(full_pointcloud),
            "full_pointcloud_normal": torch.from_numpy(full_pointcloud_normal),
            "captions": ["dummy caption"],
            "full_mesh": None,
            "category": self.category,
            "renderings": renderings,
            "raw_renderings": [],
            "meshes": [[] for _ in range(num_parts)],
            "dropped_meshes": [],
            "scales": torch.ones(num_parts, dtype=torch.float32),
        }


class DummyCragDataModule(lightning.LightningDataModule):
    """DataModule that serves synthetic batches to profile the training stack."""

    def __init__(
        self,
        *,
        train_length: int = 256,
        val_length: int = 64,
        test_length: int = 64,
        batch_size: int = 8,
        num_workers: int = 0,
        **dataset_kwargs,
    ):
        super().__init__()
        self.train_length = train_length
        self.val_length = val_length
        self.test_length = test_length
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset_kwargs = dataset_kwargs

        self.train_dataset: Dataset | None = None
        self.val_dataset: Dataset | None = None
        self.test_dataset: Dataset | None = None

    def setup(self, stage):
        if stage == "fit" or stage is None:
            self.train_dataset = DummyCragDataset(length=self.train_length, **self.dataset_kwargs)
            self.val_dataset = DummyCragDataset(length=self.val_length, **self.dataset_kwargs)

        if stage == "test" or stage is None:
            self.test_dataset = DummyCragDataset(length=self.test_length, **self.dataset_kwargs)

    def train_dataloader(self):
        assert self.train_dataset is not None
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=CragDataset.collate_fn,
        )

    def val_dataloader(self):
        assert self.val_dataset is not None
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=CragDataset.collate_fn,
        )

    def test_dataloader(self):
        assert self.test_dataset is not None
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=CragDataset.collate_fn,
        )
