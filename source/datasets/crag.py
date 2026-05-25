import gc
import io
import json
import logging
import os
from pathlib import Path
import random
import time
from typing import Literal

import h5py
import lightning
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
import torch
from torch.utils.data import Dataset
import torchvision
import tqdm
import trimesh

from ..utils.data import make_seed
from ..utils.data import pytorch_worker_seed
from ..utils.data import worker_init_fn
from ..utils.mesh import canonicalize_points
from ..utils.pcd_utils import recenter_pc
from ..utils.pcd_utils import rotate_pc
from ..utils.pcd_utils import shuffle_pc

logger = logging.getLogger("trimesh")
logger.setLevel(logging.ERROR)


class MultiCragIterableDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        *,
        split: Literal["train", "val", "test"],
        data_root: dict[str, str],
        categories: list[str],
        up_axies: dict[str, Literal["X", "Y", "Z"]] | None = None,
        category_sampling_weights: dict[str, float] | None = None,
        **kwargs,
    ):
        super().__init__()
        up_axies = up_axies or {}
        self.split = split
        self.datasets = [
            CragIterableDataset(
                split=split,
                data_root=data_root[category],
                category=category,
                up_axis=up_axies.get(category, "Y"),
                **kwargs,
            )
            for category in categories
        ]
        default_weight = 1.0
        weights = category_sampling_weights or {}
        self.dataset_weights = [weights.get(category, default_weight) for category in categories]

    def __iter__(self):
        # Worker-aware RNG so each worker shuffles categories independently
        seed = make_seed(pytorch_worker_seed(), os.getpid(), time.time_ns(), os.urandom(4))
        rng = random.Random(seed)

        iterators = [iter(dataset) for dataset in self.datasets]
        while True:
            dataset_idx = rng.choices(range(len(self.datasets)), weights=self.dataset_weights, k=1)[0]
            try:
                yield next(iterators[dataset_idx])
            except StopIteration:
                # Should not happen for train because inner dataset is infinite, but reset just in case.
                iterators[dataset_idx] = iter(self.datasets[dataset_idx])


class CragDataModule(lightning.LightningDataModule):
    def __init__(
        self,
        data_root: dict[str, str],
        categories: list[str],
        batch_size: int = 32,
        num_workers: int = 16,
        *,
        up_axies: dict[str, Literal["X", "Y", "Z"]] | None = None,
        category_sampling_weights: dict[str, float] | None = None,
        **kwargs,
    ):
        super().__init__()
        self.data_root = data_root
        self.categories = categories
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.up_axies = up_axies or {}
        self.category_sampling_weights = category_sampling_weights or {}

        self.kwargs = kwargs

        self.train_dataset: Dataset | None = None
        self.val_dataset: Dataset | None = None
        self.test_dataset: Dataset | None = None

    def setup(self, stage):
        if stage == "fit" or stage is None:
            self.train_dataset = torch.utils.data.ConcatDataset(
                [
                    CragDataset(
                        split="train",
                        data_root=self.data_root[category],
                        category=category,
                        up_axis=self.up_axies.get(category, "Y"),
                        **self.kwargs,
                    )
                    for category in self.categories
                ]
            )

            self.val_dataset = torch.utils.data.ConcatDataset(
                [
                    CragDataset(
                        split="val",
                        data_root=self.data_root[category],
                        category=category,
                        up_axis=self.up_axies.get(category, "Y"),
                        **self.kwargs,
                    )
                    for category in self.categories
                ]
            )

        if stage == "test" or stage is None:
            self.test_dataset = torch.utils.data.ConcatDataset(
                [
                    CragDataset(
                        split="test",
                        data_root=self.data_root[category],
                        category=category,
                        up_axis=self.up_axies.get(category, "Y"),
                        **self.kwargs,
                    )
                    for category in self.categories
                ]
            )

    def train_dataloader(self):
        assert self.train_dataset is not None
        # sampler = None
        # if self.train_sample_weights is not None:
        #     sampler = torch.utils.data.WeightedRandomSampler(
        #         weights=self.train_sample_weights,
        #         num_samples=len(self.train_sample_weights),
        #         replacement=True,
        #     )

        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=CragDataset.collate_fn,
            prefetch_factor=4,
            worker_init_fn=worker_init_fn,
        )

    def val_dataloader(self):
        assert self.val_dataset is not None
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=CragDataset.collate_fn,
            worker_init_fn=worker_init_fn,
        )

    def test_dataloader(self):
        assert self.test_dataset is not None
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=CragDataset.collate_fn,
            worker_init_fn=worker_init_fn,
        )


class CragDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        *,
        split: Literal["train", "val", "test"] = "train",
        data_root: str = "data.hdf5",
        category: str = "everyday",
        up_axis: Literal["X", "Y", "Z"] = "Y",
        min_parts: int = 2,
        max_parts: int = 20,
        num_points_per_object: int = 5000,
        num_points_per_part: int = 1000,
        min_points_per_part: int = 20,
        sample_by: Literal["area", "volume", "vertices", "constant"] = "area",
        multi_ref: bool = True,
        random_rotation: bool = True,
        random_anchor: bool = False,
        scale_part_pointcloud: bool = False,
        pieces_drop_rate: float = 0.0,
        pieces_drop_rate_val: float = 0.0,
        canonicalize: bool = True,
        enable_missing_from_meta: bool = False,
    ):
        super().__init__()
        self.split = split
        self.data_root = data_root
        self.category = category
        self.up_axis = up_axis
        self.min_parts = min_parts
        self.max_parts = max_parts
        self.num_points_per_object = num_points_per_object
        self.num_points_per_part = num_points_per_part
        self.min_points_per_part = min_points_per_part
        self.sample_by = sample_by
        self.multi_ref = multi_ref and self.split == "train"
        self.random_rotation = random_rotation
        self.random_anchor = random_anchor
        self.scale_part_pointcloud = scale_part_pointcloud
        self.pieces_drop_rate = pieces_drop_rate
        self.pieces_drop_rate_val = pieces_drop_rate_val
        self.canonicalize = canonicalize
        self.enable_missing_from_meta = enable_missing_from_meta
        self.missing_meta = {}
        if self.enable_missing_from_meta:
            metafile_path = data_root.replace(".hdf5", "_missing_meta.json")
            assert Path(metafile_path).is_file(), f"Missing meta file {metafile_path} not found."
            with Path(metafile_path).open("r") as f:
                self.missing_meta = json.load(f)

        # Image Transform
        self.image_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize((224, 224)),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        self.data_list = self.get_datalist()

    def get_datalist(self):
        """
        Return the list of data samples.
        """
        split = self.split
        if split == "test":
            split = "val"

        h5_file = h5py.File(self.data_root, "r")
        data_list = [d.decode("utf-8") for d in h5_file["data_split"][self.category][split]]
        filtered_data_list = []
        for item in data_list:
            try:
                num_parts = len(h5_file[item]["pieces"].keys())
                if self.min_parts <= num_parts <= self.max_parts:
                    filtered_data_list.append(item)
            except Exception:
                continue

        h5_file.close()

        return filtered_data_list

    def sample_points(
        self,
        meshes: list[trimesh.Trimesh],
    ):
        areas = [mesh.area for mesh in meshes]
        volumes = [mesh.volume for mesh in meshes]
        vertices = [len(mesh.vertices) for mesh in meshes]
        anchor_idx = np.argmax(areas)

        # Sample points based on areas, volumes, vertices, or constant
        sample_weights = np.array(
            vertices if self.sample_by == "vertices" else volumes if self.sample_by == "volume" else areas,
            dtype=np.float32,
        )
        sample_weights /= np.sum(sample_weights)

        points_per_part = [
            self.min_points_per_part
            + int((self.num_points_per_object - self.min_points_per_part * len(meshes)) * sample_weight)
            for sample_weight in sample_weights
        ]
        points_per_part[anchor_idx] += self.num_points_per_object - sum(points_per_part)
        points_per_part[np.argmin(points_per_part)] += self.num_points_per_object - np.sum(points_per_part)

        if self.sample_by == "constant":
            points_per_part = [self.num_points_per_part] * len(meshes)

        sampled_pcds = []
        for i in range(len(meshes)):
            pcd, idx = trimesh.sample.sample_surface(meshes[i], count=points_per_part[i])
            sampled_pcds.append((pcd, idx))
        pointclouds_gt = [pcd[0] for pcd in sampled_pcds]
        pointclouds_normals_gt = [meshes[i].face_normals[pcd[1]] for i, pcd in enumerate(sampled_pcds)]
        return pointclouds_gt, pointclouds_normals_gt, anchor_idx, sample_weights

    def get_data(self, name: str):
        h5_file = h5py.File(self.data_root, "r")
        pieces_names = h5_file[name]["pieces_names"][:]
        pieces_names = [name.decode("utf-8") for name in pieces_names]
        if not all(piece_name in h5_file[name]["pieces"] for piece_name in pieces_names):
            pieces_names = list(h5_file[name]["pieces"].keys())

        if self.enable_missing_from_meta:
            missing_parts = self.missing_meta.get(name, {}).get("missing_parts", [])
            if self.enable_missing_from_meta and len(missing_parts) > 0:
                pieces_names = [pn for pn in pieces_names if pn not in missing_parts]

        num_parts = len(pieces_names)
        meshes = [
            trimesh.Trimesh(
                vertices=np.array(h5_file[name]["pieces"][piece]["vertices"][:]),
                faces=np.array(h5_file[name]["pieces"][piece]["faces"][:]),
            )
            for piece in pieces_names
        ]

        if sum([mesh.area for mesh in meshes]) < 1e-6:
            # print(f"Warning: {name} has zero area, resampling...")
            h5_file.close()
            return self.get_data(np.random.choice(self.data_list))

        full_mesh_key = f"{name}/full_mesh"
        full_mesh = trimesh.Trimesh(
            vertices=np.array(h5_file[full_mesh_key]["vertices"][:]),
            faces=np.array(h5_file[full_mesh_key]["faces"][:]),
        )

        # Convert to RHS coordinate system with Y up
        if self.up_axis == "X":
            rot = trimesh.transformations.rotation_matrix(np.radians(-90), [0, 1, 0])
        elif self.up_axis == "Y":
            rot = np.eye(4)
        else:  # Z up
            rot = trimesh.transformations.rotation_matrix(np.radians(90), [1, 0, 0])
        for mesh in meshes:
            mesh.apply_transform(rot)
        full_mesh.apply_transform(rot)

        # Option to apply a random rotation to the entire object
        init_rot = Rotation.identity().as_quat()
        if self.random_rotation:
            init_rot = Rotation.random()
            for mesh in meshes:
                mesh.vertices = (init_rot.as_matrix() @ mesh.vertices.T).T
            # do not rotate full mesh to keep consistency
            full_mesh.vertices = (init_rot.as_matrix() @ full_mesh.vertices.T).T
            init_rot = init_rot.as_quat()

        # Normalize meshes, aligns with breaking bad [-0.5, 0.5]
        meshes_max_scale = float("-inf")
        for i in range(num_parts):
            abs_bounds = abs(meshes[i].bounds).flatten()
            meshes_max_scale = max([meshes_max_scale, *abs_bounds])
        meshes: list[trimesh.Trimesh] = [mesh.apply_scale(0.5 / meshes_max_scale) for mesh in meshes]
        full_mesh = full_mesh.apply_scale(0.5 / meshes_max_scale)

        # Get renderings
        renderings = []
        if "renderings" in h5_file[name]:
            rendering_keys = sorted(h5_file[name]["renderings"].keys())
            for key in rendering_keys:
                img_bytes = h5_file[name]["renderings"][key][:]
                img_pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                renderings.append(img_pil)

        drop_rate = self.pieces_drop_rate if self.split == "train" else self.pieces_drop_rate_val
        # Random drop pieces during training
        # Keep at least 2 pieces to avoid degenerate cases
        drop_pieces = sum([random.random() < drop_rate for _ in range(num_parts - 2)])
        random.shuffle(meshes)
        dropped_meshes = meshes[:drop_pieces]
        meshes = meshes[drop_pieces:]
        num_parts = len(meshes)

        pointclouds_gt, pointclouds_normals_gt, anchor_idx, sample_weights = self.sample_points(meshes=meshes)
        full_mesh_pcd, full_mesh_pcd_idx = trimesh.sample.sample_surface(full_mesh, self.num_points_per_object * 4)
        full_mesh_pcd_normal = full_mesh.face_normals[full_mesh_pcd_idx]

        h5_file.close()

        return {
            "category": self.category,
            "split": self.split,
            "name": name,
            "num_parts": num_parts,
            "pointclouds_gt": pointclouds_gt,
            "pointclouds_normals_gt": pointclouds_normals_gt,
            "anchor_idx": anchor_idx,
            "sample_weights": sample_weights,
            "pieces": ",".join(pieces_names),
            "init_rot": init_rot,
            "full_pointcloud": full_mesh_pcd,
            "full_pointcloud_normal": full_mesh_pcd_normal,
            "full_mesh": full_mesh,
            "renderings": renderings,
            "meshes": meshes,
            "dropped_meshes": dropped_meshes,
        }

    def transform(self, data: dict) -> dict:
        """
        Transform the data sample.
        Args:
            data: Data sample.
            raw_meshes: List of raw meshes.
        Returns:
            Transformed data sample.
        """
        num_parts = data["num_parts"]
        pointclouds_gt = data["pointclouds_gt"]
        pointclouds_normals_gt = data["pointclouds_normals_gt"]

        # Shuffle parts
        part_order = np.arange(num_parts)
        # np.random.shuffle(part_order)
        pointclouds_gt = [pointclouds_gt[i] for i in part_order]
        pointclouds_normals_gt = [pointclouds_normals_gt[i] for i in part_order]
        anchor_idx = np.where(part_order == data["anchor_idx"])[0][0]

        points_per_part = np.array([len(pc) for pc in pointclouds_gt])  # (valid_P,)
        offset = np.concatenate([[0], np.cumsum(points_per_part)])  # (valid_P+1,)

        pointclouds_gt = np.concatenate(pointclouds_gt)  # (N, 3)
        pointclouds_normals_gt = np.concatenate(pointclouds_normals_gt)  # (N, 3)

        # Ref-part
        ref_part = np.zeros((num_parts), dtype=np.float32)
        ref_part_idx = anchor_idx
        if self.random_anchor:
            # only points > 5% of the total points can be the ref_part
            can_be_anchor = data["sample_weights"] > 0.05
            # sample a ref part
            ref_part_idx = np.random.choice(np.where(can_be_anchor)[0], 1, replace=False)[0]

        ref_part[ref_part_idx] = 1
        ref_part = ref_part.astype(bool)

        pointclouds, pointclouds_normals, quaternions, translations, scales = [], [], [], [], []
        for part_idx in range(num_parts):
            start = offset[part_idx]
            end = offset[part_idx + 1]

            if not self.canonicalize:
                pointcloud, translation = recenter_pc(pointclouds_gt[start:end])
                pointcloud, pointcloud_normals, quaternion = rotate_pc(pointcloud, pointclouds_normals_gt[start:end])
            else:
                pointcloud, pointcloud_normals, rotation_matrix, translation = canonicalize_points(
                    pointclouds_gt[start:end], pointclouds_normals_gt[start:end]
                )
                quaternion = Rotation.from_matrix(rotation_matrix).as_quat()[
                    [3, 0, 1, 2]
                ]  # convert to wxyz to align with pytorch3d
            pointcloud, pointcloud_normals, order = shuffle_pc(pointcloud, pointcloud_normals)

            # Shuffle gt as well
            pointclouds_gt[start:end] = pointclouds_gt[start:end][order]
            pointclouds_normals_gt[start:end] = pointclouds_normals_gt[start:end][order]

            # Scale part pointcloud
            if self.scale_part_pointcloud:
                part_max_dist = np.max(np.linalg.norm(pointcloud, axis=1))
                scale = part_max_dist / 0.95
                pointcloud /= scale
            else:
                scale = 1.0

            scales.append(scale)
            pointclouds.append(pointcloud)
            pointclouds_normals.append(pointcloud_normals)
            quaternions.append(quaternion)
            translations.append(translation)

        # full_pointcloud, _ = recenter_pc(data["full_pointcloud"])
        full_pointcloud = data["full_pointcloud"]

        # Concatenate
        pointclouds = np.concatenate(pointclouds).astype(np.float32)  # [N, 3]
        pointclouds_normals = np.concatenate(pointclouds_normals).astype(np.float32)
        quaternions = np.stack(quaternions).astype(np.float32)  # [P, 4]
        translations = np.stack(translations).astype(np.float32)  # [P, 3]
        scales = np.stack(scales).astype(np.float32)  # [P,]

        if self.multi_ref and num_parts > 2 and np.random.rand() > 1 / num_parts:
            can_be_ref = data["sample_weights"] > 0.05
            can_be_ref[ref_part_idx] = False
            can_be_ref_num = np.sum(can_be_ref)
            if can_be_ref_num > 0:
                # random select more ref parts
                num_more_ref = np.random.randint(1, min(can_be_ref_num + 1, num_parts - 1))
                more_ref_part_idx = np.random.choice(np.where(can_be_ref)[0], num_more_ref, replace=False)
                ref_part[more_ref_part_idx] = True

        # Renderings
        renderings = [self.image_transform(img) for img in data["renderings"]]
        renderings = torch.stack(renderings, dim=0)  # (V, 3, 224, 224)

        out_data = {
            "num_parts": num_parts,
            "pointclouds": torch.tensor(pointclouds, dtype=torch.float32),
            "pointclouds_gt": torch.tensor(pointclouds_gt, dtype=torch.float32),
            "pointclouds_normals": torch.tensor(pointclouds_normals, dtype=torch.float32),
            "pointclouds_normals_gt": torch.tensor(pointclouds_normals_gt, dtype=torch.float32),
            "quaternions": torch.tensor(quaternions, dtype=torch.float32),
            "translations": torch.tensor(translations, dtype=torch.float32),
            "points_per_part": torch.tensor(points_per_part, dtype=torch.int32),
            "points_per_object": torch.tensor(self.num_points_per_object, dtype=torch.int32),
            "ref_part": torch.tensor(ref_part, dtype=torch.bool),
            "init_rot": torch.tensor(data["init_rot"], dtype=torch.float32),
            "full_pointcloud": torch.tensor(full_pointcloud, dtype=torch.float32),
            "full_pointcloud_normal": torch.tensor(data["full_pointcloud_normal"], dtype=torch.float32),
            "renderings": renderings,
            "scales": torch.tensor(scales, dtype=torch.float32),
        }

        if self.split != "train":
            extra_info = {
                "name": data["name"],
                "full_mesh": data["full_mesh"],
                "meshes": data["meshes"],
                "dropped_meshes": data["dropped_meshes"],
                "raw_renderings": data["renderings"],
                "category": self.category,
                "pieces": data["pieces"],
            }
            out_data.update(extra_info)

        return out_data

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        name = self.data_list[index]
        data = self.get_data(name)
        return self.transform(data)

    @staticmethod
    def collate_fn(batch):
        if not batch:
            return {}

        keys = batch[0].keys()
        ret = {}
        cat_keys = {
            "pointclouds",
            "pointclouds_gt",
            "pointclouds_normals",
            "pointclouds_normals_gt",
            "quaternions",
            "translations",
            "scales",
            "ref_part",
            "points_per_part",
            "full_pointcloud",
            "full_pointcloud_normal",
            "init_rot",
        }

        stack_keys = {"renderings", "points_per_object"}

        for k in keys:
            entries = [item[k] for item in batch]
            if k in cat_keys:
                ret[k] = torch.cat(entries, dim=0)
            elif k in stack_keys:
                ret[k] = torch.stack(entries, dim=0)
            elif k == "num_parts":
                ret[k] = torch.tensor(entries, dtype=torch.int32)
            else:
                ret[k] = entries

        return ret


class CragIterableDataset(torch.utils.data.IterableDataset, CragDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __iter__(self):
        seed = make_seed(pytorch_worker_seed(), os.getpid(), time.time_ns(), os.urandom(4))
        rng = random.Random(seed)

        total_num = 0
        failed_num = 0

        while True:
            idx = rng.randint(0, len(self.data_list) - 1)
            data_name = self.data_list[idx]

            total_num += 1
            if total_num % 1000 == 0 and failed_num > 0:
                print(f"[{self.category}] Failure rate: {failed_num}/{total_num} = {failed_num / total_num:.4f}")

            try:
                sample = self.get_data(data_name)
                sample = self.transform(sample)
                yield sample

                if total_num % 100 == 0:
                    # Force garbage collection every 100 samples to prevent memory leaks
                    gc.collect()

            except Exception:
                failed_num += 1
                continue


if __name__ == "__main__":
    splits = ["train"]
    data_roots = {
        # "breaking_bad": {
        #     "path": "/local_data/public/CRAG/BreakingBad/breaking_bad.hdf5",
        #     "categories": ["everyday", "artifact"],
        # },
        "PartNeXt": {
            "path": "/dev/shm/PartNeXt.hdf5",
            "categories": ["PartNeXt"],
        },
    }

    data_root_map = {
        category: data_info["path"] for data_info in data_roots.values() for category in data_info["categories"]
    }

    train_dataset = MultiCragIterableDataset(
        split="train",
        data_root=data_root_map,
        categories=list(data_root_map.keys()),
        num_points_per_object=20000,
        min_points_per_part=100,
        num_points_per_part=4096,
        sample_by="constant",
        random_rotation=True,
        random_anchor=False,
        min_parts=2,
        max_parts=20,
    )

    dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=4 * 8,
        shuffle=False,
        num_workers=96,
        collate_fn=CragDataset.collate_fn,
        prefetch_factor=64,
    )

    from tqdm import tqdm

    for _ in tqdm(dataloader):
        pass

    for _ in tqdm(dataloader):
        pass
