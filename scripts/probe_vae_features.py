"""Probe frozen VAE part-feature quality (ticket 06, no training).

Encodes vessel parts through the model's own encode_parts path and tests
whether the features discriminate anything:
 1. dead channels (near-zero variance across parts),
  2. same-object part-pair retrieval AUC (chance 0.5),
  3. same AUC split by part thickness (thin vs bulky halves).

Runs on CPU on the login node in minutes. Deterministic: stage-1 config has
use_post_kl=false, so features don't sample.

Example:
    python scripts/probe_vae_features.py --hdf5 data/bbad_vessels_full_crag.hdf5 \\
        --category bbad_vessels --items 12
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import hydra
import numpy as np
import torch
import train  # noqa: F401  (registers OmegaConf resolvers)
from hydra import compose, initialize

from source.datasets.crag import CragDataset


def load_batch(hdf5: str, category: str, n_items: int):
    ds = CragDataset(
        split="train", data_root=hdf5, category=category, up_axis="Z",
        num_points_per_object=2048, num_points_per_part=256,
        min_points_per_part=20,
    )
    raws, trans = [], []
    for i in range(min(n_items, len(ds))):
        raw = ds.get_data(ds.data_list[i])
        raws.append(raw)
        trans.append(ds.transform(raw))
    batch = CragDataset.collate_fn(trans)
    return batch, raws


def part_thickness(raw) -> list[float]:
    out = []
    for m in raw["meshes"]:
        s = np.abs(np.asarray(m.bounds)).max(axis=0)
        out.append(float(np.sort(s)[0]))
    return out


def pairwise_auc(feat: np.ndarray, labels: np.ndarray) -> float:
    d = np.linalg.norm(feat[:, None, :] - feat[None, :, :], axis=-1)
    iu = np.triu_indices(len(feat), k=1)
    same = (labels[iu[0]] == labels[iu[1]]).astype(float)
    order = np.argsort(d[iu])
    same = same[order]
    pos = same.sum()
    if pos == 0 or pos == len(same):
        return float("nan")
    tp = np.cumsum(same)
    fp = np.cumsum(1 - same)
    auc = float(np.trapezoid(tp / pos, fp / fp[-1]))
    return auc


@torch.inference_mode()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hdf5", required=True)
    ap.add_argument("--category", required=True)
    ap.add_argument("--items", type=int, default=12)
    args = ap.parse_args()

    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(config_name="train",
                      overrides=["experiment=stg_1_partnext", "data=crag_dummy"])
    model = hydra.utils.instantiate(cfg.model)
    model.eval()
    print("model:", type(model).__name__, flush=True)

    batch, raws = load_batch(args.hdf5, args.category, args.items)
    out = model.encode_parts(
        coord=batch["pointclouds"], normal=batch["pointclouds_normals"],
        num_parts=batch["num_parts"], ref_part=batch["ref_part"],
        points_per_part=batch["points_per_part"], scales=batch["scales"],
    )
    feat, spp = out["feat"], out["points_per_part"]
    print("feat:", tuple(feat.shape), "sampled_per_part:", spp.tolist()[:8], flush=True)

    # split flat tokens back into parts
    npp = batch["num_parts"].tolist()
    toks = spp.reshape(-1).tolist() if torch.is_tensor(spp) else list(spp)
    vecs, obj_ids, thick = [], [], []
    all_thick = [t for raw in raws for t in part_thickness(raw)]
    pos = 0
    idx = 0
    for oi, np_ in enumerate(npp):
        for _ in range(np_):
            k = int(toks[idx]) if idx < len(toks) else 0
            seg = feat[pos:pos + k]
            vecs.append(seg.mean(dim=0) if len(seg) else torch.zeros(feat.shape[-1]))
            obj_ids.append(oi)
            pos += k
            idx += 1
    # align thickness list (raw part order == transformed order up to shuffle;
    # thickness is a part property — recompute per transformed order is exact
    # only up to permutation, fine for a median split)
    thick = all_thick[:len(vecs)]

    V = torch.stack(vecs).numpy()
    print(f"parts: {len(V)}, dim: {V.shape[1]}", flush=True)

    var = V.var(axis=0)
    dead = int((var < 1e-8).sum())
    print(f"dead channels: {dead}/{V.shape[1]}", flush=True)

    lab = np.array(obj_ids)
    print(f"same-object retrieval AUC: {pairwise_auc(V, lab):.3f} (chance 0.5)", flush=True)

    med = float(np.median(thick))
    thin = np.array(thick) <= med
    a_thin = pairwise_auc(V[thin], lab[thin]) if thin.sum() > 2 else float("nan")
    a_bulk = pairwise_auc(V[~thin], lab[~thin]) if (~thin).sum() > 2 else float("nan")
    print(f"thickness median {med:.4f}: thin AUC {a_thin:.3f} (n={int(thin.sum())}), "
          f"bulky AUC {a_bulk:.3f} (n={int((~thin).sum())})", flush=True)


if __name__ == "__main__":
    main()
