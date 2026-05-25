from pytorch3d.loss.chamfer import chamfer_distance
from pytorch3d.transforms import quaternion_apply
import torch
from torch.nn.utils.rnn import pad_sequence


def scatter_mean(
    per_data_metric: torch.Tensor,  # (seq_len,)
    num_parts: torch.Tensor,  # (n_objects,)
):
    per_object_metric = torch.zeros_like(
        num_parts,
        dtype=per_data_metric.dtype,
        device=per_data_metric.device,
    )
    indices = torch.repeat_interleave(
        torch.arange(len(num_parts), device=num_parts.device),
        num_parts,
    )
    per_object_metric.scatter_add_(
        0,
        indices,
        per_data_metric,
    )
    per_object_metric /= num_parts.float()
    return per_object_metric  # (n_objects,)


def calc_part_acc(
    pcd: torch.Tensor,  # (seq_len, 3)
    num_parts: torch.Tensor,  # (n_objects,)
    points_per_part: torch.Tensor,  # (n_parts,)
    gt_transform: torch.Tensor,  # (n_parts, 7)
    pred_transform: torch.Tensor,  # (n_parts, 7)
    thre=0.01,
    scales: torch.Tensor | None = None,  # (n_parts,)
):
    gt_transform_broadcasted = gt_transform.repeat_interleave(
        points_per_part,
        dim=0,
    )  # (seq_len, 7)
    pred_transform_broadcasted = pred_transform.repeat_interleave(
        points_per_part,
        dim=0,
    )  # (seq_len, 7)
    if scales is not None:
        scales_broadcasted = scales.repeat_interleave(
            points_per_part,
            dim=0,
        ).unsqueeze(-1)  # (seq_len, 1)
        pcd = pcd * scales_broadcasted  # (seq_len, 3)

    points_per_object = torch.zeros_like(
        num_parts,
        dtype=points_per_part.dtype,
        device=points_per_part.device,
    )
    indices = torch.repeat_interleave(
        torch.arange(len(num_parts), device=num_parts.device),
        num_parts,
    )
    points_per_object.scatter_add_(
        0,
        indices,
        points_per_part,
    )  # (n_objects,)

    pcd_pred = (
        quaternion_apply(
            pred_transform_broadcasted[:, 3:],
            pcd,
        )
        + pred_transform_broadcasted[:, :3]
    )  # (seq_len, 3)

    pcd_gt = (
        quaternion_apply(
            gt_transform_broadcasted[:, 3:],
            pcd,
        )
        + gt_transform_broadcasted[:, :3]
    )  # (seq_len, 3)

    pcd_pred_per_object = torch.split(
        pcd_pred,
        points_per_object.tolist(),
        dim=0,
    )
    pcd_gt_per_object = torch.split(
        pcd_gt,
        points_per_object.tolist(),
        dim=0,
    )
    pcd_pred_per_object = pad_sequence(pcd_pred_per_object, batch_first=True)  # (n_objects, max_points, 3)
    pcd_gt_per_object = pad_sequence(pcd_gt_per_object, batch_first=True)  # (n_objects, max_points, 3)

    pcd_pred_per_part = torch.split(
        pcd_pred,
        points_per_part.tolist(),
        dim=0,
    )
    pcd_gt_per_part = torch.split(
        pcd_gt,
        points_per_part.tolist(),
        dim=0,
    )
    pcd_pred_per_part = pad_sequence(pcd_pred_per_part, batch_first=True)  # (n_parts, max_points, 3)
    pcd_gt_per_part = pad_sequence(pcd_gt_per_part, batch_first=True)  # (n_parts, max_points, 3)ƒ

    chamfer_per_data, _ = chamfer_distance(
        x=pcd_pred_per_part.float(),
        y=pcd_gt_per_part.float(),
        x_lengths=points_per_part.long(),
        y_lengths=points_per_part.long(),
        single_directional=False,
        point_reduction="mean",
        batch_reduction=None,
    )

    shape_chamfer, _ = chamfer_distance(
        x=pcd_pred_per_object.float(),
        y=pcd_gt_per_object.float(),
        x_lengths=points_per_object.long(),
        y_lengths=points_per_object.long(),
        single_directional=False,
        point_reduction="mean",
        batch_reduction=None,
    )

    return scatter_mean(
        (chamfer_per_data < thre).float(),
        num_parts=num_parts,
    ), shape_chamfer
