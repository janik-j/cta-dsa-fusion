"""Mesh I/O and processing for the clinical DSA pipeline.

OBJ read/write, visual-hull-based mesh filtering,
and uniform surface sampling.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.ndimage import binary_dilation

from data_generation.clinical.utils.geometry import ViewMaskGeometry


# ---------------------------------------------------------------------------
# OBJ I/O
# ---------------------------------------------------------------------------


def load_obj(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load a simple OBJ file.

    Returns ``(vertices [N, 3], faces [M, 3])`` with 0-based indices.
    Polygons with >3 vertices are fan-triangulated.
    """
    verts: list[list[float]] = []
    faces: list[list[int]] = []

    with open(path, encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("v "):
                parts = line.split()
                if len(parts) >= 4:
                    verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
                continue
            if line.startswith("f "):
                parts = line.split()[1:]
                if len(parts) < 3:
                    continue
                idxs: list[int] = []
                for p in parts:
                    s = p.split("/")[0]
                    if s:
                        idxs.append(int(s) - 1)
                if len(idxs) < 3:
                    continue
                a = idxs[0]
                for j in range(1, len(idxs) - 1):
                    faces.append([a, idxs[j], idxs[j + 1]])

    return (
        np.asarray(verts, dtype=np.float32).reshape(-1, 3),
        np.asarray(faces, dtype=np.int64).reshape(-1, 3),
    )


def write_obj(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    """Write ``(vertices [N, 3], faces [M, 3])`` to OBJ (1-based face indices)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for v in np.asarray(vertices, dtype=np.float32):
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        faces_1 = np.asarray(faces, dtype=np.int64) + 1
        for tri in faces_1:
            f.write(f"f {int(tri[0])} {int(tri[1])} {int(tri[2])}\n")


# ---------------------------------------------------------------------------
# Visual hull mesh filtering
# ---------------------------------------------------------------------------


def filter_mesh_by_visual_hull(
    *,
    mesh_vertices_world_xyz: np.ndarray,
    mesh_faces: np.ndarray,
    masks_with_geometry: list[ViewMaskGeometry],
    mask_dilate_px: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Remove mesh vertices (and their triangles) not visible in all views.

    Each vertex is projected into every view's 2-D mask.  Vertices not
    visible in all masks are discarded, along with any face that
    references a discarded vertex.
    """
    verts = np.asarray(mesh_vertices_world_xyz, dtype=np.float32)
    faces = np.asarray(mesh_faces, dtype=np.int64)

    n_views = len(masks_with_geometry)
    if n_views < 1:
        raise ValueError("masks_with_geometry must contain at least one view")

    n = int(verts.shape[0])
    votes = np.zeros(n, dtype=np.uint8)

    for view in masks_with_geometry:
        mask_2d = view.mask_2d
        if int(mask_dilate_px) > 0:
            mask_2d = binary_dilation(mask_2d.astype(bool), iterations=int(mask_dilate_px))
        mask_2d = mask_2d.astype(bool, copy=False)

        h, w = mask_2d.shape
        vec = verts - view.cam_pos[None, :]
        cam_x = np.sum(vec * view.cam_right[None, :], axis=1)
        cam_y = np.sum(vec * view.cam_up[None, :], axis=1)
        cam_z = np.sum(vec * view.cam_forward[None, :], axis=1)

        in_front = cam_z > 0.0
        u = view.fx * cam_x / (cam_z + 1e-8) + view.cx
        v = view.fy * cam_y / (cam_z + 1e-8) + view.cy
        u_int = np.rint(u).astype(np.int32)
        v_int = np.rint(v).astype(np.int32)

        in_bounds = in_front & (u_int >= 0) & (u_int < w) & (v_int >= 0) & (v_int < h)
        hit = np.zeros(n, dtype=bool)
        if np.any(in_bounds):
            hit[in_bounds] = mask_2d[v_int[in_bounds], u_int[in_bounds]]
        votes += hit.astype(np.uint8)

    keep_v = votes >= n_views
    keep_f = keep_v[faces].all(axis=1)
    faces_kept = faces[keep_f]

    if faces_kept.size == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.int64)

    used = np.unique(faces_kept.ravel())
    remap = -np.ones(n, dtype=np.int64)
    remap[used] = np.arange(used.shape[0], dtype=np.int64)
    return verts[used], remap[faces_kept]


# ---------------------------------------------------------------------------
# Island removal
# ---------------------------------------------------------------------------


def remove_mesh_islands(
    *,
    vertices_world_xyz: np.ndarray,
    faces: np.ndarray,
    min_size_ratio: float = 0.01,
) -> tuple[np.ndarray, np.ndarray]:
    """Remove small disconnected components from a triangle mesh.

    Components are identified via Union-Find on shared edges.
    Any component smaller than ``min_size_ratio`` of the largest
    component (by vertex count) is discarded.
    """
    verts = np.asarray(vertices_world_xyz, dtype=np.float32)
    tri = np.asarray(faces, dtype=np.int64)
    if tri.size == 0 or verts.size == 0:
        return verts.reshape((-1, 3)), tri.reshape((-1, 3))

    n = int(verts.shape[0])
    parent = np.arange(n, dtype=np.int64)
    rank = np.zeros(n, dtype=np.int8)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return int(x)

    def union(a: int, b: int) -> None:
        ra, rb = find(int(a)), find(int(b))
        if ra == rb:
            return
        if rank[ra] < rank[rb]:
            parent[ra] = rb
        elif rank[ra] > rank[rb]:
            parent[rb] = ra
        else:
            parent[rb] = ra
            rank[ra] += 1

    a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
    for i in range(tri.shape[0]):
        union(a[i], b[i])
        union(b[i], c[i])
        union(c[i], a[i])

    used = np.unique(tri.ravel())
    roots = np.array([find(int(i)) for i in used], dtype=np.int64)
    uniq, counts = np.unique(roots, return_counts=True)
    if uniq.size == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.int64)

    max_count = int(counts.max())
    min_keep = max(1, int(np.floor(float(max_count) * float(min_size_ratio))))
    keep_roots = set(int(r) for r, c0 in zip(uniq, counts) if int(c0) >= min_keep)

    root_of = np.full((n,), -1, dtype=np.int64)
    for v in used:
        root_of[int(v)] = find(int(v))

    tri_roots = root_of[tri]
    keep_faces_mask = np.isin(tri_roots, list(keep_roots)).all(axis=1)
    tri_kept = tri[keep_faces_mask]
    if tri_kept.size == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.int64)

    used2 = np.unique(tri_kept.ravel())
    remap = -np.ones((n,), dtype=np.int64)
    remap[used2] = np.arange(used2.shape[0], dtype=np.int64)
    return verts[used2], remap[tri_kept]


# ---------------------------------------------------------------------------
# Surface sampling
# ---------------------------------------------------------------------------


def sample_points_on_mesh(
    *,
    vertices_world_xyz: np.ndarray,
    faces: np.ndarray,
    n_points: int,
    seed: int = 0,
) -> np.ndarray:
    """Uniformly sample *n_points* on a triangle mesh surface (Turk 1990)."""
    verts = np.asarray(vertices_world_xyz, dtype=np.float32)
    tri = np.asarray(faces, dtype=np.int64)
    n_points = int(n_points)
    if n_points <= 0 or tri.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float32)

    v0, v1, v2 = verts[tri[:, 0]], verts[tri[:, 1]], verts[tri[:, 2]]
    areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
    total = float(areas.sum())
    if not np.isfinite(total) or total <= 0:
        rng = np.random.default_rng(int(seed))
        idx = rng.choice(verts.shape[0], size=min(n_points, verts.shape[0]), replace=False)
        return verts[idx]

    rng = np.random.default_rng(int(seed))
    tri_idx = rng.choice(tri.shape[0], size=n_points, replace=True, p=areas / total)
    a, b, c = v0[tri_idx], v1[tri_idx], v2[tri_idx]

    r1 = rng.random(n_points, dtype=np.float32)
    r2 = rng.random(n_points, dtype=np.float32)
    sqrt_r1 = np.sqrt(r1)
    u = 1.0 - sqrt_r1
    v = sqrt_r1 * (1.0 - r2)
    w = sqrt_r1 * r2
    return (a * u[:, None] + b * v[:, None] + c * w[:, None]).astype(np.float32)
