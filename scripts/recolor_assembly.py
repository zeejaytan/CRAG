"""
Bake per-part macaron colors into the multi-method assembly comparison GLBs.

For every shape under ``static/assembly/`` we have one GLB per method:
    static/assembly/{gt, crag, assembler, garf, rpf}/<shape_id>.glb

Each GLB is a trimesh.Scene whose geometries are the same physical parts —
but methods may order their geometries differently in the file (in particular,
CRAG permutes parts relative to GT/Assembler/GARF/RPF). Within every row the
vertex counts are all-distinct, so vertex count uniquely identifies a part.

We therefore:
  1. Load GT and build a canonical mapping  vertex_count -> palette index.
  2. For every method (including GT), assign palette[canonical_idx(vertex_count)]
     to each geometry. Same physical part -> same color across all 5 methods.

Outputs are written next to the originals as ``<shape_id>_colored.glb`` so the
originals stay untouched.

Usage:
    python3 scripts/recolor_assembly.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh
from trimesh.visual.material import PBRMaterial
from trimesh.visual import TextureVisuals


# Pastel "macaron" palette as RGBA floats in [0, 1]. Kept identical to the
# palette in scripts/recolor_partnext.py so colors look uniform across the
# teaser section and the comparison grid.
MACARON = [
    (0.965, 0.620, 0.680, 1.0),  # rose
    (0.580, 0.855, 0.745, 1.0),  # sage mint
    (0.700, 0.730, 0.910, 1.0),  # periwinkle
    (0.985, 0.770, 0.640, 1.0),  # salmon peach
    (0.975, 0.855, 0.530, 1.0),  # honey
    (0.600, 0.815, 0.965, 1.0),  # cornflower
    (0.830, 0.650, 0.860, 1.0),  # orchid
    (0.970, 0.510, 0.590, 1.0),  # strawberry coral
    (0.670, 0.870, 0.620, 1.0),  # pistachio
    (0.970, 0.660, 0.860, 1.0),  # bubblegum
    (0.540, 0.790, 0.790, 1.0),  # teal
    (0.975, 0.705, 0.575, 1.0),  # apricot
]


REPO_ROOT = Path(__file__).resolve().parents[1]
ASSEMBLY_ROOT = REPO_ROOT / "static" / "assembly"

# Anchor method whose part order defines the canonical color assignment.
ANCHOR_METHOD = "gt"
METHODS = ["gt", "crag", "assembler", "garf", "rpf"]


def make_material(color_idx: int, tag: str) -> PBRMaterial:
    r, g, b, a = MACARON[color_idx % len(MACARON)]
    return PBRMaterial(
        name=f"{tag}_part_{color_idx}",
        baseColorFactor=[r, g, b, a],
        metallicFactor=0.0,
        roughnessFactor=0.85,
        doubleSided=True,
    )


def ordered_geometry_keys(scene: trimesh.Scene) -> list[str]:
    """Geometry keys sorted by their trailing integer (geometry_0, geometry_1, ...)."""

    def key_idx(k: str) -> tuple[int, str]:
        tail = k.rsplit("_", 1)[-1]
        try:
            return (0, f"{int(tail):010d}")
        except ValueError:
            return (1, k)

    return sorted(scene.geometry.keys(), key=key_idx)


def part_fingerprint(geom: trimesh.Trimesh) -> tuple:
    """Stable, similarity-invariant fingerprint for a part.

    Two GLBs may contain the same physical part placed differently *and* at
    different scales — methods sometimes re-normalize parts before exporting
    (RPF rescales by ~3×, Assembler nudges scale by a few percent). Raw
    covariance eigenvalues drift with scale. We therefore normalize the
    sorted eigenvalues of the centered vertex covariance by their sum, which
    yields a fingerprint invariant under rotation, translation, AND uniform
    scale.

    Vertex count is included as the leading element since two distinct parts
    that happen to share a shape-ratio are almost always disambiguated by
    vertex count.
    """
    verts = np.asarray(geom.vertices, dtype=np.float64)
    n = int(len(verts))
    if n < 2:
        return (n,)
    centered = verts - verts.mean(axis=0)
    cov = (centered.T @ centered) / n
    eigs = np.linalg.eigvalsh(cov)  # ascending, real, length 3 for 3D
    total = float(eigs.sum())
    if total <= 0:
        return (n,)  # degenerate (all vertices coincident)
    ratios = tuple(round(float(e) / total, 3) for e in eigs)
    return (n,) + ratios


def canonical_fingerprint_map(anchor_path: Path) -> list[tuple]:
    """Ordered list of GT part fingerprints; index in the list is the color slot.

    Returned as a *list* (not a dict) because duplicate fingerprints can occur
    when a shape has truly identical parts (e.g. four identical chair legs).
    Lookup uses :func:`assign_colors_against_canonical` which consumes each
    canonical slot at most once per method file.
    """
    scene = trimesh.load(anchor_path, force=None)
    if not isinstance(scene, trimesh.Scene):
        scene = trimesh.Scene(geometry=scene)
    return [part_fingerprint(scene.geometry[k]) for k in ordered_geometry_keys(scene)]


def assign_colors_against_canonical(
    method_fps: list[tuple], canonical_fps: list[tuple]
) -> tuple[list[int], int]:
    """For each method part, pick the GT-slot index whose fingerprint matches.

    Slots are claimed greedily: once a method part claims canonical slot j,
    that slot is unavailable for other method parts. Unmatched method parts
    fall back to their own running index modulo len(palette), so they still
    get a distinct color even if no GT correspondence exists.

    Returns (color_indices, matched_count).
    """
    available = list(range(len(canonical_fps)))
    out: list[int] = []
    matched = 0
    for i, fp in enumerate(method_fps):
        chosen = None
        for j in available:
            if canonical_fps[j] == fp:
                chosen = j
                break
        if chosen is None:
            out.append(i % len(MACARON))
        else:
            available.remove(chosen)
            out.append(chosen)
            matched += 1
    return out, matched


def recolor_method_file(
    src: Path, dst: Path, canonical_fps: list[tuple], tag: str
) -> tuple[int, int]:
    scene = trimesh.load(src, force=None)
    if not isinstance(scene, trimesh.Scene):
        scene = trimesh.Scene(geometry=scene)

    keys = ordered_geometry_keys(scene)
    method_fps = [part_fingerprint(scene.geometry[k]) for k in keys]
    color_indices, matched = assign_colors_against_canonical(method_fps, canonical_fps)

    for k, color_idx in zip(keys, color_indices):
        material = make_material(color_idx, tag)
        try:
            scene.geometry[k].visual = TextureVisuals(material=material)
        except Exception:
            scene.geometry[k].visual.material = material  # type: ignore[attr-defined]

    scene.export(dst)
    return matched, len(keys)


def main() -> None:
    if not ASSEMBLY_ROOT.exists():
        raise SystemExit(f"missing directory: {ASSEMBLY_ROOT}")

    anchor_dir = ASSEMBLY_ROOT / ANCHOR_METHOD
    shape_files = sorted(p.name for p in anchor_dir.glob("*.glb") if "_colored" not in p.stem)
    if not shape_files:
        raise SystemExit(f"no GLBs under {anchor_dir}")

    print(f"anchor method = {ANCHOR_METHOD}  shapes = {len(shape_files)}\n")

    for shape_name in shape_files:
        print(f"shape: {shape_name}")
        anchor_path = anchor_dir / shape_name
        canonical_fps = canonical_fingerprint_map(anchor_path)
        print(f"  canonical parts: {len(canonical_fps)}")

        for method in METHODS:
            src = ASSEMBLY_ROOT / method / shape_name
            if not src.exists():
                print(f"  skip (missing): {src.relative_to(REPO_ROOT)}")
                continue
            dst = src.with_name(src.stem + "_colored.glb")
            tag = f"{method}_{src.stem}"
            matched, total = recolor_method_file(src, dst, canonical_fps, tag)
            warn = "" if matched == total else "  [partial — non-matched parts use fallback colors]"
            print(f"  {method:10s} -> {dst.relative_to(REPO_ROOT)}  "
                  f"({matched}/{total} parts matched canonically){warn}")
        print()


if __name__ == "__main__":
    main()
