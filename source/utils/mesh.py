import numpy as np
from sklearn.decomposition import PCA
import trimesh


def bbox(points):
    try:
        to_origin, size = trimesh.bounds.oriented_bounds(obj=points, angle_digits=1)
        center = to_origin[:3, :3].transpose().dot(-to_origin[:3, 3])

        xdir = to_origin[0, :3]
        ydir = to_origin[1, :3]
        zdir = to_origin[2, :3]
    except Exception:
        points = np.array(points)
        pca = PCA()
        pca.fit(points)
        pcomps = pca.components_
        points_local = np.matmul(pcomps, points.transpose()).transpose()
        all_max = points_local.max(axis=0)
        all_min = points_local.min(axis=0)
        center = np.dot(np.linalg.inv(pcomps), (all_max + all_min) / 2)
        size = all_max - all_min
        xdir = pcomps[0, :]
        xdir /= np.linalg.norm(xdir)
        ydir = pcomps[1, :]
        ydir /= np.linalg.norm(ydir)
        zdir = np.cross(xdir, ydir)
        zdir /= np.linalg.norm(zdir)

    R = np.vstack([xdir, ydir, zdir]).transpose().astype(np.float32)
    t = center.astype(np.float32)
    return R, t, size


def canonicalize_mesh(mesh: trimesh.Trimesh, *, return_transform=False):
    points = mesh.sample(1000)
    R, t, _ = bbox(points)

    v = mesh.vertices
    part_v = np.dot(v - t, np.linalg.inv(R).transpose())

    part_mesh = trimesh.Trimesh(vertices=part_v, faces=mesh.faces)
    if return_transform:
        return part_mesh, R, t
    return part_mesh


def canonicalize_points(points: np.ndarray, normals: np.ndarray | None = None):
    R, t, _ = bbox(points)
    points_canonical = np.dot(points - t, np.linalg.inv(R).transpose())
    normals_canonical = None
    if normals is not None:
        normals_canonical = np.dot(normals, np.linalg.inv(R).transpose())
    return points_canonical, normals_canonical, R, t
