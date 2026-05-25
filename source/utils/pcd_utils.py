import random

import numpy as np
from scipy.spatial.transform import Rotation


def recenter_pc(pc):
    """pc: [N, 3]"""
    centroid = np.mean(pc, axis=0)
    return pc - centroid[None], centroid


def rotate_pc(pc, normal=None, numpy_rng=None):
    """
    pc: [N, 3]
    normal: [N, 3] or None
    """
    rot_mat = Rotation.random(random_state=numpy_rng).as_matrix()
    rotated_pc = (rot_mat @ pc.T).T
    quat_gt = Rotation.from_matrix(rot_mat.T).as_quat()
    # we use scalar-first quaternion
    quat_gt = quat_gt[[3, 0, 1, 2]]
    if normal is None:
        return rotated_pc, None, quat_gt

    rotated_normal = (rot_mat @ normal.T).T
    return rotated_pc, rotated_normal, quat_gt


def shuffle_pc(pc, normal=None):
    """pc: [N, 3]"""
    order = np.arange(pc.shape[0])
    random.shuffle(order)
    shuffled_pc = pc[order]
    if normal is None:
        return shuffled_pc, None, order

    shuffled_normal = normal[order]
    return shuffled_pc, shuffled_normal, order


def rotate_whole_part(pc, normal=None):
    """
    pc: [P, N, 3]
    """
    P, N, _ = pc.shape
    pc = pc.reshape(-1, 3)
    rot_mat = Rotation.random().as_matrix()
    pc = (rot_mat @ pc.T).T
    quat_gt = Rotation.from_matrix(rot_mat.T).as_quat()
    # we use scalar-first quaternion
    quat_gt = quat_gt[[3, 0, 1, 2]]
    if normal is None:
        return pc.reshape(P, N, 3), None, quat_gt
    rotated_normal = (rot_mat @ normal.reshape(-1, 3).T).T.reshape(P, N, 3)
    return pc.reshape(P, N, 3), rotated_normal, quat_gt


def perturb_pc(
    *,
    pc,
    normal=None,
    trans_perturb=0.01,
    rot_perturb=0.01,
):
    """pc: [N, 3]"""
    # Random translation
    translation = np.random.normal(scale=trans_perturb, size=(1, 3))

    # Random small rotation
    axis = np.random.normal(size=3)
    axis /= np.linalg.norm(axis) + 1e-8  # normalize
    angle = np.random.normal(scale=rot_perturb)
    rot = Rotation.from_rotvec(axis * angle)
    rot_matrix = rot.as_matrix()

    perturbed_pc = (pc @ rot_matrix.T) + translation

    if normal is not None:
        assert normal.shape == pc.shape, "normal must have the same shape as pc"
        perturbed_normal = normal @ rot_matrix.T
        return perturbed_pc, perturbed_normal

    return perturbed_pc, None
