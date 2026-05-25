"""
This script will generate missing metadata files for our processed hdf5 datasets.
"""

import dataclasses
import json
from multiprocessing import Pool
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm
import trimesh
import tyro

"""
Missing Logic:
1. # of parts missed = min(floor(total # of parts * missing_percent), max(total # of parts - 2, 0))
2. Do not throw away the largest part, judged by bounding box volume.
3. Randomly select the parts to be removed.

Example: 34%
2: 0
3: 1
4: 1
5: 1
6: 2
7: 2
8: 2
9: 3
10: 3
11: 3
12: 4
...
"""


@dataclasses.dataclass
class Config:
    seed: int = 42
    data_path: Path = Path("/local_data/public/CRAG/final_data/skull.hdf5")
    output_path: Path = Path("/local_data/public/CRAG/final_data/skull_missing_meta.json")
    num_processes: int = 64
    missing_percent: float = 0.34


def process_item(args):
    """Process a single item and determine which parts to remove."""
    item_id, h5_path, missing_percent, seed = args

    with h5py.File(h5_path, "r") as f:
        if item_id not in f:
            return item_id, None

        if "pieces" not in f[item_id]:
            return item_id, None

        pieces_keys = list(f[item_id]["pieces"].keys())
        num_parts = len(pieces_keys)
        num_missing = min(int(num_parts * missing_percent), max(num_parts - 2, 0))

        if num_missing == 0:
            return item_id, {"num_parts": num_parts, "num_missing": 0, "missing_parts": []}

        # Calculate bounding box volume for each part
        part_volumes = {}
        for part_key in pieces_keys:
            part_data = f[item_id]["pieces"][part_key]
            if "vertices" in part_data and "faces" in part_data:
                try:
                    vertices = np.array(part_data["vertices"][:])
                    faces = np.array(part_data["faces"][:])
                    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
                    # Use mesh volume if available, otherwise use bounding box volume
                    volume = mesh.volume if mesh.is_watertight else mesh.bounding_box.volume
                    part_volumes[part_key] = volume
                except Exception:
                    # Fallback to bounding box volume if mesh construction fails
                    vertices = np.array(part_data["vertices"][:])
                    bbox_min = vertices.min(axis=0)
                    bbox_max = vertices.max(axis=0)
                    volume = np.prod(bbox_max - bbox_min)
                    part_volumes[part_key] = volume
            else:
                part_volumes[part_key] = 0.0

        # Find the largest part (do not remove it)
        largest_part = max(part_volumes.items(), key=lambda x: x[1])[0]

        # Randomly select parts to remove (excluding the largest part)
        rng = np.random.RandomState(seed + hash(item_id) % (2**31))
        removable_parts = [k for k in pieces_keys if k != largest_part]

        num_missing = min(num_missing, len(removable_parts))

        missing_parts = rng.choice(removable_parts, size=num_missing, replace=False).tolist()

        return item_id, {
            "num_parts": num_parts,
            "num_missing": num_missing,
            "missing_parts": missing_parts,
            "largest_part": largest_part,
        }


def main(cfg: Config):
    print(f"Processing dataset: {cfg.data_path}")
    print(f"Missing percent: {cfg.missing_percent * 100}%")
    print(f"Random seed: {cfg.seed}")
    print(f"Number of processes: {cfg.num_processes}")

    # Collect all items to process
    with h5py.File(cfg.data_path, "r") as f:
        all_items = []
        if "data_split" in f:
            for category in f["data_split"]:
                for split in f["data_split"][category]:
                    items = [item.decode("utf-8") for item in f["data_split"][category][split][:]]
                    all_items.extend(items)
        else:
            # If no data_split structure, process all top-level items
            all_items = list(f.keys())

    print(f"Total items to process: {len(all_items)}")

    # Prepare arguments for multiprocessing
    args_list = [(item_id, str(cfg.data_path), cfg.missing_percent, cfg.seed) for item_id in all_items]

    # Process items with multiprocessing
    metadata = {}
    with Pool(processes=cfg.num_processes) as pool:
        results = list(tqdm(pool.imap(process_item, args_list), total=len(args_list), desc="Processing items"))

    # Collect results
    metadata = {k: v for k, v in results if v is not None}

    # Save metadata to JSON
    cfg.output_path.parent.mkdir(parents=True, exist_ok=True)
    with cfg.output_path.open("w") as f:
        json.dump(metadata, f)

    print(f"\nMetadata saved to: {cfg.output_path}")
    print(f"Total items processed: {len(metadata)}")

    # Print statistics
    total_parts = sum(m["num_parts"] for m in metadata.values())
    total_missing = sum(m["num_missing"] for m in metadata.values())
    print(f"Total parts: {total_parts}")
    print(f"Total missing parts: {total_missing}")
    if total_parts > 0:
        print(f"Actual missing rate: {total_missing / total_parts * 100:.2f}%")


if __name__ == "__main__":
    main(tyro.cli(Config))
