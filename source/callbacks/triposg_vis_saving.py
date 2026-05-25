import json
from pathlib import Path

import lightning as L
import trimesh


class TripoSGVisualizationCallback(L.Callback):
    """Saves TripoSG benchmark outputs: input image, generated GLB, and per-sample metrics JSON."""

    def __init__(self, *, save_dir: str | None = None):
        super().__init__()
        self.save_dir = save_dir

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if outputs is None:
            return

        save_path = (
            Path(self.save_dir) if self.save_dir is not None else Path(trainer.default_root_dir) / "visualizations"
        )

        batch_size = len(outputs["name"])
        view_index = pl_module.view_index

        for i in range(batch_size):
            name = outputs["name"][i]
            obj_path = save_path / name
            obj_path.mkdir(parents=True, exist_ok=True)

            # Save input image used for generation.
            batch["raw_renderings"][i][view_index].save(obj_path / f"view_{view_index}.png")

            # Save generated mesh as GLB.
            gen_mesh: trimesh.Trimesh = outputs["meshes"][i]
            gen_mesh.export(obj_path / f"view_{view_index}_generated.glb")

            # Save GT full mesh as GLB for visual comparison.
            gt_mesh: trimesh.Trimesh = batch["full_mesh"][i]
            gt_mesh.export(obj_path / "view_gt.glb")

            # Save metrics JSON.
            stats = {"shape_chamfer": outputs["shape_chamfer"][i].item() if outputs["shape_chamfer"][i] is not None else None}
            with (obj_path / f"view_{view_index}.json").open("w") as f:
                json.dump(stats, f, indent=2)
