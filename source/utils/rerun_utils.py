import itertools

import rerun as rr
import torch


def log_pointcloud(
    name: str | list[str],
    pointcloud: torch.Tensor,
    colors: tuple[int, int, int] | None = None,
):
    rr.log(
        name,
        rr.Points3D(
            positions=pointcloud.cpu().numpy(),
            colors=colors,
        ),
    )


def log_pointclouds_seq(
    name: str | list[str],
    pointclouds: torch.Tensor,
    sequence_length: torch.Tensor,
):
    cum_sequence_length = torch.cumsum(sequence_length, dim=0)
    cum_sequence_length = torch.cat(
        (torch.tensor([0], device=cum_sequence_length.device), cum_sequence_length[:-1]),
        dim=0,
    )

    if isinstance(name, list):
        name = "/".join(name)

    for i in range(len(sequence_length)):
        start = cum_sequence_length[i]
        end = start + sequence_length[i]
        log_pointcloud(
            name=f"{name}/{i}",
            pointcloud=pointclouds[start:end],
        )


def log_pose_result(
    num_parts: torch.Tensor,  # (num_objects,)
    points_per_part: torch.Tensor,  # (num_parts,)
    transforms: torch.Tensor,  # (steps, num_parts, 7)
    pointclouds: torch.Tensor,  # (N, 3)
    pointclouds_gt: torch.Tensor,  # (N, 3)
    name: list[str] | None = None,  # (num_objects,)
    batch_idx: int | None = None,
):
    assert name is not None or batch_idx is not None, "Either name or batch_idx must be provided"
    batch_size = len(num_parts)
    if name is None:
        name = [f"{batch_size * batch_idx + i}" for i in range(batch_size)]

    cum_parts = torch.cumsum(num_parts, dim=0)
    cum_parts = torch.cat(
        (torch.tensor([0], device=cum_parts.device), cum_parts),
        dim=0,
    )
    cum_points_per_part = torch.cumsum(points_per_part, dim=0)
    cum_points_per_part = torch.cat(
        (torch.tensor([0], device=cum_points_per_part.device), cum_points_per_part),
        dim=0,
    )

    rr.set_time("step_idx", sequence=0)
    # log all pointclouds first
    for object_idx, (start_part, end_part) in enumerate(itertools.pairwise(cum_parts)):
        for part in range(start_part, end_part):
            log_pointcloud(
                name=f"{name[object_idx]}/pointclouds/{part - start_part}",
                pointcloud=pointclouds[cum_points_per_part[part] : cum_points_per_part[part + 1]],
            )
            log_pointcloud(
                name=f"{name[object_idx]}/pointclouds_gt/{part - start_part}",
                pointcloud=pointclouds_gt[cum_points_per_part[part] : cum_points_per_part[part + 1]],
            )

    # log all transforms
    for step_idx, transform in enumerate(transforms):
        rr.set_time("step_idx", sequence=step_idx + 1)
        for object_idx, (start_part, end_part) in enumerate(itertools.pairwise(cum_parts)):
            for part in range(start_part, end_part):
                rr.log(
                    f"{name[object_idx]}/pointclouds/{part - start_part}",
                    rr.Transform3D(
                        translation=transform[part, :3].cpu().numpy(),
                        quaternion=transform[part, 3:][[1, 2, 3, 0]].cpu().numpy(),  # wxyz to xyzw conversion
                    ),
                )
