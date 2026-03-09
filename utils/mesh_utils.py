"""Mesh helpers for evaluation (OBJ I/O + mesh extraction + distance metrics)."""

from __future__ import annotations

import math
import os
from typing import Any

import numpy as np


def _require_mcubes():
    try:
        import mcubes  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "mcubes failed to import. Mesh extraction requires `PyMCubes` built against the active NumPy. "
            "Install the base repo env (`uv sync`) and ensure user-site packages don't shadow it "
            "(e.g. set `PYTHONNOUSERSITE=1`)."
        ) from e
    return mcubes


def _require_ckdtree():
    try:
        from scipy.spatial import cKDTree  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError("scipy.spatial.cKDTree is required for mesh metric computation.") from e
    return cKDTree


def read_obj_vertices(path: str) -> np.ndarray:
    """Read vertex positions from an OBJ file."""
    verts: list[list[float]] = []
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("v "):
                parts = line.strip().split()
                if len(parts) >= 4:
                    verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
    arr = np.asarray(verts, dtype=np.float64)
    # Ensure consistent shape even for empty meshes.
    if arr.size == 0:
        return np.zeros((0, 3), dtype=np.float64)
    return arr


def transform_obj_vertices(in_path: str, out_path: str, R: np.ndarray, t: np.ndarray) -> None:
    """Apply a rigid transform to OBJ vertex lines and write a new OBJ."""
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    t = np.asarray(t, dtype=np.float64).reshape(3)
    tmp_path = out_path + ".tmp"
    with open(in_path, encoding="utf-8", errors="ignore") as fin, open(tmp_path, "w", encoding="utf-8") as fout:
        for line in fin:
            if line.startswith("v "):
                parts = line.strip().split()
                if len(parts) >= 4:
                    v = np.array([float(parts[1]), float(parts[2]), float(parts[3])], dtype=np.float64)
                    w = (R @ v) + t
                    fout.write(f"v {w[0]:.8f} {w[1]:.8f} {w[2]:.8f}\n")
                else:
                    fout.write(line)
            else:
                fout.write(line)
    os.replace(tmp_path, out_path)


def _best_fit_rigid_transform(A: np.ndarray, B: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute best-fit rigid transform (R, t) that maps A -> B in least squares sense.

    A, B: (N, 3) arrays with known correspondences.
    """
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    if A.shape != B.shape or A.ndim != 2 or A.shape[1] != 3:
        raise ValueError(f"Expected A and B to be (N,3) with same shape, got A={A.shape}, B={B.shape}")
    if A.shape[0] < 3:
        raise ValueError("Need at least 3 points for rigid transform.")

    centroid_A = A.mean(axis=0)
    centroid_B = B.mean(axis=0)
    AA = A - centroid_A
    BB = B - centroid_B

    H = AA.T @ BB
    U, _S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    # Handle reflection.
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    t = centroid_B - (R @ centroid_A)
    return R, t


def icp_rigid_align(
    source: np.ndarray,
    target: np.ndarray,
    *,
    max_iterations: int = 75,
    tolerance: float = 1e-6,
    sample_size: int = 2000,
    seed: int = 0,
    trim_fraction: float = 0.8,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Simple point-to-point ICP to rigidly align `source` onto `target`.

    Returns (R, t, final_mean_error) such that aligned ≈ (R @ source.T).T + t.

    Notes:
    - Deterministic sampling (seed) for reproducible evaluation.
    - Uses a trimmed correspondence set (default 80%) for robustness.
    """
    from scipy.spatial import cKDTree  # type: ignore

    src = np.asarray(source, dtype=np.float64).reshape(-1, 3)
    tgt = np.asarray(target, dtype=np.float64).reshape(-1, 3)
    if src.shape[0] == 0 or tgt.shape[0] == 0:
        return np.eye(3, dtype=np.float64), np.zeros((3,), dtype=np.float64), float("inf")

    k = int(min(int(sample_size), int(src.shape[0])))
    if k < 3:
        return np.eye(3, dtype=np.float64), np.zeros((3,), dtype=np.float64), float("inf")

    trim_fraction = float(trim_fraction)
    if not (0.1 <= trim_fraction <= 1.0):
        raise ValueError(f"trim_fraction must be in [0.1, 1.0], got {trim_fraction}")

    rng = np.random.default_rng(int(seed))
    sample_idx = rng.choice(src.shape[0], size=k, replace=False)

    tree = cKDTree(tgt)

    R_total = np.eye(3, dtype=np.float64)
    t_total = np.zeros((3,), dtype=np.float64)
    src_aligned = src.copy()

    prev_err = math.inf
    for _it in range(int(max_iterations)):
        pts = src_aligned[sample_idx]
        dists, nn_idx = tree.query(pts, k=1, workers=-1)
        dists = np.asarray(dists, dtype=np.float64).reshape(-1)
        nn_idx = np.asarray(nn_idx, dtype=np.int64).reshape(-1)
        matched = tgt[nn_idx]

        # Trim out worst correspondences.
        if trim_fraction < 1.0:
            keep = int(max(3, math.floor(trim_fraction * len(dists))))
            order = np.argpartition(dists, keep - 1)[:keep]
            pts_use = pts[order]
            matched_use = matched[order]
            err = float(dists[order].mean())
        else:
            pts_use = pts
            matched_use = matched
            err = float(dists.mean())

        if not np.isfinite(err):
            break

        R_delta, t_delta = _best_fit_rigid_transform(pts_use, matched_use)
        src_aligned = (src_aligned @ R_delta.T) + t_delta  # apply to full set

        # Compose transforms (new ∘ old).
        R_total = R_delta @ R_total
        t_total = (R_delta @ t_total) + t_delta

        if abs(prev_err - err) < float(tolerance):
            prev_err = err
            break
        prev_err = err

    return R_total, t_total, float(prev_err)


def icp_align_obj(
    *,
    mesh_ref_path: str,
    mesh_pred_path: str,
    mesh_pred_aligned_path: str,
    max_iterations: int = 75,
    sample_size: int = 2000,
    seed: int = 0,
    trim_fraction: float = 0.8,
) -> dict[str, Any]:
    """
    Align `mesh_pred_path` onto `mesh_ref_path` with rigid ICP and write aligned OBJ.

    Returns a small dict with alignment diagnostics.
    """
    ref_v = read_obj_vertices(mesh_ref_path)
    pred_v = read_obj_vertices(mesh_pred_path)
    if ref_v.shape[0] == 0 or pred_v.shape[0] == 0:
        # Still write-through for reproducibility/debugging.
        transform_obj_vertices(mesh_pred_path, mesh_pred_aligned_path, np.eye(3), np.zeros((3,)))
        return {"ok": False, "mean_error": float("inf"), "n_ref": int(ref_v.shape[0]), "n_pred": int(pred_v.shape[0])}

    R, t, err = icp_rigid_align(
        pred_v,
        ref_v,
        max_iterations=int(max_iterations),
        sample_size=int(sample_size),
        seed=int(seed),
        trim_fraction=float(trim_fraction),
    )
    transform_obj_vertices(mesh_pred_path, mesh_pred_aligned_path, R, t)
    return {
        "ok": bool(np.isfinite(err)),
        "mean_error": float(err),
        "n_ref": int(ref_v.shape[0]),
        "n_pred": int(pred_v.shape[0]),
    }


def to_spacing(spacing: Any) -> np.ndarray:
    """Convert spacing to a (1, 3) array."""
    return np.asarray(spacing, dtype=np.float64).reshape(1, 3)


def load_mesh_vertices(path: str, spacing: np.ndarray) -> np.ndarray:
    """Load OBJ vertices and scale by spacing."""
    return read_obj_vertices(path) * spacing


def nn_distances(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Nearest-neighbor distance from each point in src -> dst."""
    cKDTree = _require_ckdtree()
    tree = cKDTree(dst)
    dists, _ = tree.query(src, k=1, workers=-1)
    return np.asarray(dists, dtype=np.float64)


def chamfer_hd95(points_a: np.ndarray, points_b: np.ndarray) -> tuple[float, float]:
    """Bidirectional Chamfer and 95th Hausdorff (mm)."""
    if len(points_a) == 0 or len(points_b) == 0:
        return float("inf"), float("inf")

    d_ab = nn_distances(points_a, points_b)
    d_ba = nn_distances(points_b, points_a)
    cd = float(d_ab.mean() + d_ba.mean())
    hd95 = float(max(np.percentile(d_ab, 95), np.percentile(d_ba, 95)))
    return cd, hd95


def mesh_extraction(vol_pred: np.ndarray, thres: float, mesh_pred_path: str) -> None:
    """Extract a surface mesh from a predicted voxel volume using marching cubes."""
    mcubes = _require_mcubes()
    verts, faces = mcubes.marching_cubes(vol_pred, thres)
    mcubes.export_obj(verts, faces, mesh_pred_path)
    print("Mesh extraction done! Saved to:", mesh_pred_path)


def compute_metric(mesh_ref_path: str, mesh_pred_path: str, spacing) -> tuple[float, float]:
    """Compute CD + HD95 between two OBJ meshes (mm)."""
    spacing = to_spacing(spacing)
    verts_ref = load_mesh_vertices(mesh_ref_path, spacing)
    verts_pred = load_mesh_vertices(mesh_pred_path, spacing)
    return chamfer_hd95(verts_ref, verts_pred)


def compute_metric_missing_region(mesh_missing_path: str, mesh_pred_path: str, spacing) -> tuple[float, float]:
    """Compute one-directional metrics (GT missing -> Prediction)."""
    if not os.path.exists(mesh_missing_path):
        print(f"Warning: Missing mesh file not found at {mesh_missing_path}. Returning NaN.")
        return np.nan, np.nan

    spacing = to_spacing(spacing)
    verts_missing = load_mesh_vertices(mesh_missing_path, spacing)
    if len(verts_missing) == 0:
        print("Warning: Missing region mesh is empty. Returning 0.0.")
        return 0.0, 0.0

    verts_pred = load_mesh_vertices(mesh_pred_path, spacing)
    if len(verts_pred) == 0:
        return float("inf"), float("inf")

    d = nn_distances(verts_missing, verts_pred)
    return float(d.mean()), float(np.percentile(d, 95))


def compute_metric_missing_region_partitioned(
    mesh_missing_path: str, mesh_pred_path: str, mesh_prior_path: str, spacing
) -> tuple[float, float, float, float]:
    """Compute metrics on missing vessel region using prediction partitioning.

    Partitions prediction vertices by nearest GT region, then computes
    bidirectional metrics - directly comparable to global CD/HD95.
    """
    nan_result = (np.nan, np.nan, np.nan, np.nan)
    inf_result = (float("inf"), float("inf"), float("inf"), float("inf"))

    if not os.path.exists(mesh_missing_path):
        print(f"Warning: Missing mesh file not found at {mesh_missing_path}. Returning NaN.")
        return nan_result

    if not os.path.exists(mesh_prior_path):
        print(f"Warning: Prior mesh file not found at {mesh_prior_path}. Returning NaN.")
        return nan_result

    spacing = to_spacing(spacing)
    verts_missing = load_mesh_vertices(mesh_missing_path, spacing)
    verts_prior = load_mesh_vertices(mesh_prior_path, spacing)
    verts_pred = load_mesh_vertices(mesh_pred_path, spacing)

    if len(verts_pred) == 0:
        return inf_result

    if len(verts_missing) == 0:
        print("Warning: Missing region mesh is empty.")
        return 0.0, 0.0, np.nan, np.nan

    if len(verts_prior) == 0:
        print("Warning: Prior region mesh is empty.")
        return nan_result

    d_pred_to_missing = nn_distances(verts_pred, verts_missing)
    d_pred_to_prior = nn_distances(verts_pred, verts_prior)
    pred_missing_mask = d_pred_to_missing < d_pred_to_prior
    verts_pred_missing = verts_pred[pred_missing_mask]
    verts_pred_prior = verts_pred[~pred_missing_mask]

    if len(verts_pred_missing) == 0:
        print("Warning: No prediction points assigned to missing region.")
        cd_missing, hd_missing = float("inf"), float("inf")
    else:
        cd_missing, hd_missing = chamfer_hd95(verts_missing, verts_pred_missing)

    if len(verts_pred_prior) == 0:
        print("Warning: No prediction points assigned to prior region.")
        cd_prior, hd_prior = float("inf"), float("inf")
    else:
        cd_prior, hd_prior = chamfer_hd95(verts_prior, verts_pred_prior)

    return cd_missing, hd_missing, cd_prior, hd_prior


__all__ = [
    "compute_metric",
    "compute_metric_missing_region",
    "compute_metric_missing_region_partitioned",
    "icp_align_obj",
    "mesh_extraction",
]
