import json
from pathlib import Path

import lightning as L
import numpy as np
from pygltflib import ARRAY_BUFFER
from pygltflib import ELEMENT_ARRAY_BUFFER
from pygltflib import FLOAT
from pygltflib import GLTF2
from pygltflib import UNSIGNED_INT
from pygltflib import Accessor
from pygltflib import Animation
from pygltflib import AnimationChannel
from pygltflib import AnimationChannelTarget
from pygltflib import AnimationSampler
from pygltflib import Buffer
from pygltflib import BufferView
from pygltflib import Mesh
from pygltflib import Node
from pygltflib import Primitive
from pygltflib import Scene
import pytorch3d.transforms as p3dt
import torch
import trimesh


class VisualizationSavingCallback(L.Callback):
    def __init__(self, *, save_dir: str | None = None, animation_fps: float = 10.0):
        super().__init__()
        self.save_dir = save_dir
        self.animation_fps = animation_fps

    @staticmethod
    def se3_to_matrix(transformation: torch.Tensor) -> torch.Tensor:
        """Convert se3 transform [tx, ty, tz, qw, qx, qy, qz] to a 4x4 matrix."""
        t = transformation[:3]
        q = transformation[3:]

        R = p3dt.quaternion_to_matrix(q)
        M = torch.eye(4, device=transformation.device, dtype=transformation.dtype)
        M[:3, :3] = R
        M[:3, 3] = t
        return M

    @staticmethod
    def se3_to_trs(transformation: torch.Tensor) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Convert se3 transform [tx, ty, tz, qw, qx, qy, qz] to translation, rotation quaternion, and scale."""
        t = transformation[:3].cpu().numpy()
        q = transformation[3:].cpu().numpy()  # [qw, qx, qy, qz]
        # GLTF uses [qx, qy, qz, qw] format
        q_gltf = np.array([q[1], q[2], q[3], q[0]])
        s = np.array([1.0, 1.0, 1.0])
        return t, q_gltf, s

    def save_animation_glb(
        self,
        output_path: Path,
        meshes: list[trimesh.Trimesh],
        transform_steps: list[torch.Tensor],
        gt_transforms: torch.Tensor,
        fps: float = 10.0,
    ):
        """
        Save animation as a single GLB file with keyframes.

        Args:
            output_path: Path to save the GLB file
            meshes: List of part meshes
            transform_steps: List of transforms for each animation step, each of shape [num_parts, 7]
            gt_transforms: Ground truth transforms [num_parts, 7]
            fps: Frames per second for animation playback
        """
        gltf = GLTF2()
        gltf.scene = 0
        gltf.scenes = [Scene(nodes=[])]
        gltf.nodes = []
        gltf.meshes = []
        gltf.accessors = []
        gltf.bufferViews = []
        gltf.buffers = [Buffer()]
        gltf.animations = []

        binary_blob = bytearray()
        num_parts = len(meshes)
        num_steps = len(transform_steps)

        # Add each part mesh and create nodes
        for part_idx, part_mesh in enumerate(meshes):
            # Create mesh primitives
            vertices = part_mesh.vertices.astype(np.float32)
            faces = part_mesh.faces.astype(np.uint32)

            # Add vertices to buffer
            vertices_blob = vertices.tobytes()
            vertices_buffer_view = len(gltf.bufferViews)
            gltf.bufferViews.append(
                BufferView(
                    buffer=0,
                    byteOffset=len(binary_blob),
                    byteLength=len(vertices_blob),
                    target=ARRAY_BUFFER,
                )
            )
            binary_blob.extend(vertices_blob)

            # Add vertex accessor
            vertices_accessor = len(gltf.accessors)
            gltf.accessors.append(
                Accessor(
                    bufferView=vertices_buffer_view,
                    byteOffset=0,
                    componentType=FLOAT,
                    count=len(vertices),
                    type="VEC3",
                    min=vertices.min(axis=0).tolist(),
                    max=vertices.max(axis=0).tolist(),
                )
            )

            # Add indices to buffer
            indices_blob = faces.flatten().tobytes()
            indices_buffer_view = len(gltf.bufferViews)
            gltf.bufferViews.append(
                BufferView(
                    buffer=0,
                    byteOffset=len(binary_blob),
                    byteLength=len(indices_blob),
                    target=ELEMENT_ARRAY_BUFFER,
                )
            )
            binary_blob.extend(indices_blob)

            # Add indices accessor
            indices_accessor = len(gltf.accessors)
            gltf.accessors.append(
                Accessor(
                    bufferView=indices_buffer_view,
                    byteOffset=0,
                    componentType=UNSIGNED_INT,
                    count=len(faces) * 3,
                    type="SCALAR",
                )
            )

            # Create mesh
            mesh_idx = len(gltf.meshes)
            gltf.meshes.append(
                Mesh(
                    primitives=[
                        Primitive(
                            attributes={"POSITION": vertices_accessor},
                            indices=indices_accessor,
                        )
                    ]
                )
            )

            # Create node with initial transform (relative to GT)
            node_idx = len(gltf.nodes)
            with torch.autocast(enabled=False, device_type=gt_transforms.device.type):
                gt_transform_mat = self.se3_to_matrix(gt_transforms[part_idx].float())
                initial_transform_mat = self.se3_to_matrix(transform_steps[0][part_idx].float())
                relative_transform = initial_transform_mat @ torch.linalg.inv(gt_transform_mat)

            # Convert to TRS
            t, r, s = self.matrix_to_trs(relative_transform)

            gltf.nodes.append(
                Node(
                    mesh=mesh_idx,
                    translation=t.tolist(),
                    rotation=r.tolist(),
                    scale=s.tolist(),
                )
            )
            gltf.scenes[0].nodes.append(node_idx)

        # Create animation
        if num_steps > 1:
            animation = Animation(channels=[], samplers=[])

            # Time keyframes based on fps (0, 1/fps, 2/fps, ..., (num_steps-1)/fps, (num_steps-1)/fps + 1.0)
            # Add an extra keyframe at the end to hold the last frame for 1 second
            time_values = np.arange(num_steps, dtype=np.float32) / fps
            time_values = np.append(time_values, time_values[-1] + 1.0)  # Hold last frame for 1 second
            num_keyframes = len(time_values)

            time_blob = time_values.tobytes()
            time_buffer_view = len(gltf.bufferViews)
            gltf.bufferViews.append(
                BufferView(
                    buffer=0,
                    byteOffset=len(binary_blob),
                    byteLength=len(time_blob),
                )
            )
            binary_blob.extend(time_blob)

            time_accessor = len(gltf.accessors)
            gltf.accessors.append(
                Accessor(
                    bufferView=time_buffer_view,
                    byteOffset=0,
                    componentType=FLOAT,
                    count=num_keyframes,
                    type="SCALAR",
                    min=[0.0],
                    max=[float(time_values[-1])],
                )
            )

            # Create animation channels for each part
            for part_idx in range(num_parts):
                # Collect all transforms for this part
                translations = []
                rotations = []
                scales = []

                for step_transforms in transform_steps:
                    with torch.autocast(enabled=False, device_type=gt_transforms.device.type):
                        gt_transform_mat = self.se3_to_matrix(gt_transforms[part_idx].float())
                        step_transform_mat = self.se3_to_matrix(step_transforms[part_idx].float())
                        relative_transform = step_transform_mat @ torch.linalg.inv(gt_transform_mat)

                    t, r, s = self.matrix_to_trs(relative_transform)
                    translations.append(t)
                    rotations.append(r)
                    scales.append(s)

                # Duplicate the last frame to hold it for 1 second
                translations.append(translations[-1])
                rotations.append(rotations[-1])
                scales.append(scales[-1])

                # Add translation keyframes
                translation_values = np.array(translations, dtype=np.float32)
                translation_blob = translation_values.tobytes()
                translation_buffer_view = len(gltf.bufferViews)
                gltf.bufferViews.append(
                    BufferView(
                        buffer=0,
                        byteOffset=len(binary_blob),
                        byteLength=len(translation_blob),
                    )
                )
                binary_blob.extend(translation_blob)

                translation_accessor = len(gltf.accessors)
                gltf.accessors.append(
                    Accessor(
                        bufferView=translation_buffer_view,
                        byteOffset=0,
                        componentType=FLOAT,
                        count=num_keyframes,
                        type="VEC3",
                    )
                )

                # Add rotation keyframes
                rotation_values = np.array(rotations, dtype=np.float32)
                rotation_blob = rotation_values.tobytes()
                rotation_buffer_view = len(gltf.bufferViews)
                gltf.bufferViews.append(
                    BufferView(
                        buffer=0,
                        byteOffset=len(binary_blob),
                        byteLength=len(rotation_blob),
                    )
                )
                binary_blob.extend(rotation_blob)

                rotation_accessor = len(gltf.accessors)
                gltf.accessors.append(
                    Accessor(
                        bufferView=rotation_buffer_view,
                        byteOffset=0,
                        componentType=FLOAT,
                        count=num_keyframes,
                        type="VEC4",
                    )
                )

                # Create samplers and channels for translation and rotation
                translation_sampler_idx = len(animation.samplers)
                animation.samplers.append(
                    AnimationSampler(
                        input=time_accessor,
                        output=translation_accessor,
                        interpolation="LINEAR",
                    )
                )

                rotation_sampler_idx = len(animation.samplers)
                animation.samplers.append(
                    AnimationSampler(
                        input=time_accessor,
                        output=rotation_accessor,
                        interpolation="LINEAR",
                    )
                )

                animation.channels.append(
                    AnimationChannel(
                        sampler=translation_sampler_idx,
                        target=AnimationChannelTarget(
                            node=part_idx,
                            path="translation",
                        ),
                    )
                )

                animation.channels.append(
                    AnimationChannel(
                        sampler=rotation_sampler_idx,
                        target=AnimationChannelTarget(
                            node=part_idx,
                            path="rotation",
                        ),
                    )
                )

            gltf.animations.append(animation)

        # Set buffer data
        gltf.buffers[0].byteLength = len(binary_blob)
        gltf.set_binary_blob(bytes(binary_blob))

        # Save to file
        gltf.save(str(output_path))

    @staticmethod
    def matrix_to_trs(matrix: torch.Tensor) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Convert 4x4 transformation matrix to translation, rotation quaternion, and scale."""
        t = matrix[:3, 3].cpu().numpy()
        R = matrix[:3, :3]

        # Extract scale (assuming uniform scaling or no scaling)
        s = np.array([1.0, 1.0, 1.0])

        # Convert rotation matrix to quaternion
        # Using pytorch3d's function
        q = p3dt.matrix_to_quaternion(R.unsqueeze(0)).squeeze(0).cpu().numpy()  # [qw, qx, qy, qz]
        # GLTF uses [qx, qy, qz, qw] format
        q_gltf = np.array([q[1], q[2], q[3], q[0]])

        return t, q_gltf, s

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        return self.on_test_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)

    def on_test_batch_end(
        self,
        trainer,
        pl_module,
        outputs,
        batch,
        batch_idx,
        dataloader_idx=0,
    ):
        if trainer.testing:
            save_path = (
                Path(self.save_dir) if self.save_dir is not None else Path(trainer.default_root_dir) / "visualizations"
            )

        if trainer.validating or trainer.sanity_checking:
            save_path = (
                Path(self.save_dir) if self.save_dir is not None else Path(trainer.default_root_dir) / "visualizations"
            ) / f"step-{trainer.global_step}"
            # only save the first batch of validation for visualization
            if batch_idx > 0:
                return

        bsz = len(batch["name"])

        # Create directories for each object
        for name in batch["name"]:
            (save_path / name).mkdir(parents=True, exist_ok=True)

        # Extract outputs
        transform_0 = outputs["gt_transform"]
        transform_t = outputs["pred_transform"]
        transforms_steps = outputs["pred_transform_steps"]
        part_acc = outputs["part_acc"]
        shape_chamfer = outputs["shape_chamfer"]
        trans_rmse = outputs["rmse_t"]
        rot_rmse = outputs["rmse_r"]
        gen_meshes = outputs["meshes"]

        # Handle multi-view case
        for object_idx in range(bsz):
            object_path = save_path / batch["name"][object_idx]
            original_num_parts = batch["num_parts"][object_idx]

            # Save each view
            for view_idx, image_idx in enumerate(outputs["image_indices"][object_idx]):
                # Save rendered image if available
                if image_idx >= 0:
                    batch["raw_renderings"][object_idx][image_idx].save(object_path / f"view_{image_idx}.png")

                # Save generated meshes for this view
                mesh_idx = view_idx * bsz + object_idx
                if mesh_idx < len(gen_meshes):
                    gen_meshes[mesh_idx].export(object_path / f"view_{image_idx}.glb")

                # Create assembly scene for this view
                scene = trimesh.Scene()

                for part_idx in range(original_num_parts):
                    part_mesh: trimesh.Trimesh = batch["meshes"][object_idx][part_idx]

                    # Calculate global part index
                    part_global_idx = (
                        batch["num_parts"].sum() * view_idx + batch["num_parts"][:object_idx].sum() + part_idx
                    )

                    # Get ground truth and predicted transforms
                    part_gt_transform = transform_0[part_global_idx]
                    part_pred_transform = transform_t[part_global_idx]

                    # Convert to 4x4 transformation matrices
                    with torch.autocast(enabled=False, device_type=pl_module.device.type):
                        part_gt_transform_mat = self.se3_to_matrix(part_gt_transform.float())
                        part_pred_transform_mat = self.se3_to_matrix(part_pred_transform.float())
                        # Compute relative transform: from GT to predicted
                        T_final = part_pred_transform_mat @ torch.linalg.inv(part_gt_transform_mat)
                        scene.add_geometry(part_mesh.copy(), transform=T_final.float().cpu().numpy())

                # Save assembly scene
                scene.export(object_path / f"view_assembly_{image_idx}.glb")

                # Save animation as a single GLB file with keyframes
                animation_transforms = []
                for transforms in transforms_steps:
                    part_transforms = []
                    for part_idx in range(original_num_parts):
                        part_global_idx = (
                            batch["num_parts"].sum() * view_idx + batch["num_parts"][:object_idx].sum() + part_idx
                        )
                        part_transforms.append(transforms[part_global_idx])
                    animation_transforms.append(torch.stack(part_transforms))

                gt_part_transforms = []
                for part_idx in range(original_num_parts):
                    part_global_idx = (
                        batch["num_parts"].sum() * view_idx + batch["num_parts"][:object_idx].sum() + part_idx
                    )
                    gt_part_transforms.append(transform_0[part_global_idx])
                gt_part_transforms = torch.stack(gt_part_transforms)

                self.save_animation_glb(
                    object_path / f"view_assembly_animation_{image_idx}.glb",
                    batch["meshes"][object_idx],
                    animation_transforms,
                    gt_part_transforms,
                    fps=self.animation_fps,
                )

                # Save statistics for this view
                mesh_idx = view_idx * bsz + object_idx
                stats = {
                    "part_acc": part_acc[mesh_idx].item(),
                    "shape_chamfer": shape_chamfer[mesh_idx].item(),
                    "rmse_t": trans_rmse[mesh_idx].item(),
                    "rmse_r": rot_rmse[mesh_idx].item(),
                    "drop_pieces": len(batch.get("dropped_meshes", [[]])[object_idx])
                    if "dropped_meshes" in batch
                    else 0,
                }
                with (object_path / f"view_{image_idx}.json").open("w") as f:
                    json.dump(stats, f, indent=2)

            # Save ground truth scene (only once per object)
            scene_gt = trimesh.Scene()
            for part_idx in range(original_num_parts):
                part_mesh: trimesh.Trimesh = batch["meshes"][object_idx][part_idx]
                scene_gt.add_geometry(part_mesh)
            scene_gt.export(object_path / "view_gt.glb")

            # Save input scene (only once per object)
            scene_input = trimesh.Scene()
            for part_idx in range(original_num_parts):
                part_mesh: trimesh.Trimesh = batch["meshes"][object_idx][part_idx]
                part_global_idx = batch["num_parts"][:object_idx].sum() + part_idx
                part_gt_transform = transform_0[part_global_idx]
                with torch.autocast(enabled=False, device_type=pl_module.device.type):
                    part_gt_transform_mat = self.se3_to_matrix(part_gt_transform.float())
                    T_input = torch.linalg.inv(part_gt_transform_mat)
                    scene_input.add_geometry(part_mesh.copy(), transform=T_input.float().cpu().numpy())
            scene_input.export(object_path / "view_input.glb")
