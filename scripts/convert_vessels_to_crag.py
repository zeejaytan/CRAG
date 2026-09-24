"""Convert a pieces-only HDF5 (TORA vessels, GARF Juglet) to CRAG schema.

Adds per item: ``full_mesh`` (concatenated piece meshes, faces offset) and
``renderings`` (deterministic gray placeholder PNGs — the image condition is
dropped at train time, ``image_drop_rate=1.0``, i.e. the paper's "w/o img"
mode; placeholders exist because the datamodule stacks them unconditionally).
Rebuilds ``data_split/{category}/{split}`` for the requested splits.

Reads are from any checkout (paths are CLI args); output goes under CRAG
``data/`` (gitignored, heavy). Runs on the login node: ~hundreds of items,
minutes.

Example:
    python scripts/convert_vessels_to_crag.py \\
        --source /data/gpfs/projects/punim2657/TORA/dataset/bbad_vessels_v3.hdf5 \\
        --output data/bbad_vessels_crag.hdf5 --category bbad_vessels \\
        --splits train:bbad_vessels_v3/train_screened val:bbad_vessels_v3/val_screened
"""

import argparse
import io

import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm

GRAY = (128, 128, 128)


def placeholder_png(size: int) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (size, size), GRAY).save(buf, format="PNG")
    return buf.getvalue()


def convert_item(src_item: h5py.Group, dst_item: h5py.Group, png: bytes, n_views: int) -> int:
    names = [n.decode("utf-8") for n in src_item["pieces_names"][:]]
    if not all(n in src_item["pieces"] for n in names):
        names = list(src_item["pieces"].keys())
    dst_item.create_dataset("pieces_names", data=np.array(names, dtype="S"))

    all_v, all_f, offset = [], [], 0
    pieces = dst_item.create_group("pieces")
    for n in names:
        v = np.array(src_item["pieces"][n]["vertices"][:], dtype=np.float32)
        f = np.array(src_item["pieces"][n]["faces"][:], dtype=np.int64)
        pieces.create_dataset(f"{n}/vertices", data=v, compression="gzip", compression_opts=4)
        pieces.create_dataset(f"{n}/faces", data=f, compression="gzip", compression_opts=4)
        all_v.append(v)
        all_f.append(f + offset)
        offset += len(v)
    full = dst_item.create_group("full_mesh")
    full.create_dataset("vertices", data=np.concatenate(all_v), compression="gzip", compression_opts=4)
    full.create_dataset("faces", data=np.concatenate(all_f), compression="gzip", compression_opts=4)

    rend = dst_item.create_group("renderings")
    for i in range(n_views):
        rend.create_dataset(f"view_{i:02d}.png",
                            data=np.frombuffer(png, dtype=np.uint8),
                            compression="gzip", compression_opts=4)
    return len(names)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--category", required=True)
    ap.add_argument("--splits", nargs="+", required=True,
                    help="dst:src/cat/split triples, e.g. train:bbad_vessels_v3/train_screened")
    ap.add_argument("--views", type=int, default=12)
    ap.add_argument("--render-size", type=int, default=64)
    args = ap.parse_args()

    png = placeholder_png(args.render_size)
    with h5py.File(args.source, "r") as src, h5py.File(args.output, "w") as dst:
        for spec in args.splits:
            dst_split, src_path = spec.split(":", 1)
            names = [n.decode("utf-8") for n in src["data_split"][src_path][:]]
            out_names = []
            for name in tqdm(names, desc=f"{args.category}/{dst_split}"):
                if name not in src:
                    print(f"  skip (absent): {name}")
                    continue
                try:
                    n_parts = convert_item(src[name], dst.create_group(name), png, args.views)
                except Exception as e:  # noqa: BLE001 — log and continue; screening at read time
                    print(f"  skip (error): {name}: {type(e).__name__} {str(e)[:100]}")
                    continue
                if not 2 <= n_parts <= 20:
                    del dst[name]
                    continue
                out_names.append(name)
            dst.create_dataset(f"data_split/{args.category}/{dst_split}",
                               data=np.array(out_names, dtype="S"))
            print(f"wrote {len(out_names)}/{len(names)} items -> data_split/{args.category}/{dst_split}")


if __name__ == "__main__":
    main()
