"""Contains `sharp quest-lod` CLI implementation.

Builds a lighter level-of-detail copy of a SHARP splat file for standalone
headsets (e.g. Meta Quest 3), saved next to it as `<scene>_quest.ply`.

SHARP predicts a pixel-aligned grid of Gaussians with two layers per pixel,
which allows two reductions that generic splat pruning can't do:

1. Redundant layer pruning: where both layers of a pixel sit at nearly the same
   depth and the front one is opaque, the back one is invisible from any nearby
   viewpoint and is dropped.
2. Adaptive quadtree merging: in each layer, 2x2 blocks of neighbouring
   Gaussians that are all present, with similar depth, colour and opacity,
   are replaced by one Gaussian with matching mean/covariance (moment
   matching). This repeats on the merged grid, so flat regions collapse to 4x4
   or 8x8 blocks while edges and texture keep full resolution.

For licensing see accompanying LICENSE file.
"""

from __future__ import annotations

import logging
from pathlib import Path

import click
import numpy as np
from plyfile import PlyData, PlyElement

from sharp.utils import logging as logging_utils

LOGGER = logging.getLogger(__name__)

SH_C0 = 0.28209479177387814
PROPS = ["x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
         "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]


def quat_to_rot(q: np.ndarray) -> np.ndarray:
    """(N,4) wxyz quaternions -> (N,3,3) rotation matrices."""
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    w, x, y, z = q.T
    return np.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ], axis=1).reshape(-1, 3, 3)


def rot_to_quat(R: np.ndarray) -> np.ndarray:
    """(N,3,3) rotation matrices -> (N,4) wxyz quaternions."""
    m00, m11, m22 = R[:, 0, 0], R[:, 1, 1], R[:, 2, 2]
    cands = np.stack([
        np.stack([1 + m00 + m11 + m22, R[:, 2, 1] - R[:, 1, 2], R[:, 0, 2] - R[:, 2, 0], R[:, 1, 0] - R[:, 0, 1]], 1),
        np.stack([R[:, 2, 1] - R[:, 1, 2], 1 + m00 - m11 - m22, R[:, 0, 1] + R[:, 1, 0], R[:, 0, 2] + R[:, 2, 0]], 1),
        np.stack([R[:, 0, 2] - R[:, 2, 0], R[:, 0, 1] + R[:, 1, 0], 1 - m00 + m11 - m22, R[:, 1, 2] + R[:, 2, 1]], 1),
        np.stack([R[:, 1, 0] - R[:, 0, 1], R[:, 0, 2] + R[:, 2, 0], R[:, 1, 2] + R[:, 2, 1], 1 - m00 - m11 + m22], 1),
    ], 1)
    best = np.argmax(np.stack([m00 + m11 + m22, m00, m11, m22], 1), axis=1)
    q = cands[np.arange(len(R)), best]
    return q / np.linalg.norm(q, axis=1, keepdims=True)


def cov_to_scale_rot(cov: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(N,3,3) covariances -> (N,3) log-scales, (N,4) wxyz quaternions."""
    evals, evecs = np.linalg.eigh(cov)
    flip = np.linalg.det(evecs) < 0
    evecs[flip, :, 2] *= -1
    return 0.5 * np.log(np.maximum(evals, 1e-12)), rot_to_quat(evecs)


def to_raw(mu, cov, rgb, alpha) -> np.ndarray:
    """Gaussian parameters -> (N,14) rows in PROPS order."""
    log_scale, quat = cov_to_scale_rot(cov)
    dc = (rgb - 0.5) / SH_C0
    alpha = np.clip(alpha, 1e-4, 1 - 1e-4)
    logit = np.log(alpha / (1 - alpha))
    return np.concatenate([mu, dc, logit[:, None], log_scale, quat], 1).astype(np.float32)


def build_lod(raw: np.ndarray, grid: int, args) -> np.ndarray:
    n_layers = raw.shape[0] // (grid * grid)
    raw = raw.reshape(n_layers, grid, grid, 14)
    alpha = 1 / (1 + np.exp(-raw[..., 6]))
    depth = raw[..., 2]

    # 1. Redundant layer pruning: drop the back splat of a co-located pair when
    # the front one is opaque.
    present = np.ones((n_layers, grid, grid), dtype=bool)
    if n_layers == 2:
        front_is_0 = depth[0] <= depth[1]
        rel_gap = np.abs(depth[1] - depth[0]) / np.maximum(np.minimum(depth[0], depth[1]), 1e-6)
        front_alpha = np.where(front_is_0, alpha[0], alpha[1])
        redundant = (rel_gap < args.layer_gap) & (front_alpha > args.min_alpha)
        present[1] &= ~(redundant & front_is_0)
        present[0] &= ~(redundant & ~front_is_0)
        LOGGER.info("Layer pruning removed %d splats.", int((~present).sum()))

    outputs = []
    for layer in range(n_layers):
        r = raw[layer].reshape(-1, 14)
        p = present[layer]
        # Level-0 cells are emitted from the original rows (no round-trip error);
        # merged cells are tracked as Gaussian parameters.
        mu = r[:, :3].astype(np.float64)
        rot = quat_to_rot(r[:, 10:14].astype(np.float64))
        s2 = np.exp(2 * r[:, 7:10].astype(np.float64))
        cov = rot @ (s2[:, :, None] * rot.transpose(0, 2, 1))
        rgb = 0.5 + SH_C0 * r[:, 3:6].astype(np.float64)
        a = alpha[layer].reshape(-1).astype(np.float64)
        level_raw = r
        size = grid

        for level in range(args.levels + 1):
            if level == args.levels or size % 2:
                outputs.append(level_raw[p.reshape(-1)])
                break
            h = size // 2

            def blocks(x):  # (size*size, ...) -> (h*h, 4, ...)
                x = x.reshape(h, 2, h, 2, *x.shape[1:]).swapaxes(1, 2)
                return x.reshape(h * h, 4, *x.shape[4:])

            bp, bmu, bcov, brgb, ba = map(blocks, (p.reshape(-1), mu, cov, rgb, a))
            bz = bmu[..., 2]
            merge = (
                bp.all(1)
                & (ba.max(1) - ba.min(1) < args.alpha_tol)
                & ((bz.max(1) - bz.min(1)) / np.maximum(np.abs(bz).mean(1), 1e-6) < args.depth_tol)
                & ((brgb.max(1) - brgb.min(1)).max(1) < args.color_tol)
            )
            # Cells in non-merged blocks are emitted at this level.
            keep = bp & ~merge[:, None]
            keep_grid = keep.reshape(h, h, 2, 2).swapaxes(1, 2).reshape(-1)
            outputs.append(level_raw[keep_grid])

            # Moment-match each merged block (equal weights).
            m = bmu.mean(1)
            d = bmu - m[:, None]
            cov = bcov.mean(1) + np.einsum("bki,bkj->bij", d, d) / 4
            mu, rgb, a = m, brgb.mean(1), ba.mean(1)
            p = merge
            size = h
            level_raw = to_raw(mu, cov, rgb, a) if p.any() else np.zeros((h * h, 14), np.float32)
            LOGGER.info("Layer %d level %d: merged %d blocks of 4.", layer, level + 1, int(merge.sum()))

    return np.concatenate(outputs, 0)


def process(path: Path, args) -> None:
    out_path = path.with_name(f"{path.stem}_quest.ply")
    if out_path.exists() and not args.overwrite:
        LOGGER.info("%s exists, skipping.", out_path)
        return
    ply = PlyData.read(path)
    v = ply["vertex"].data
    raw = np.stack([v[k] for k in PROPS], 1).astype(np.float32)
    grid = int(ply["image_size"].data["image_size"][0]) if "image_size" in ply else 0
    n = len(raw)
    # SHARP stores layers of a square grid; fall back to inferring it.
    if not grid or n % (grid * grid):
        grid = int(round(np.sqrt(n / 2)))
        if 2 * grid * grid != n:
            LOGGER.warning("%s is not a 2-layer square grid, skipping.", path)
            return
    lod = build_lod(raw, grid, args)
    vertex = np.empty(len(lod), dtype=[(k, "f4") for k in PROPS])
    for i, k in enumerate(PROPS):
        vertex[k] = lod[:, i]
    PlyData([PlyElement.describe(vertex, "vertex")]).write(out_path)
    LOGGER.info("%s: %d -> %d splats (%.1f%%), saved %s", path.name, n, len(lod), 100 * len(lod) / n, out_path)


@click.command()
@click.option("-i", "--input-path", type=click.Path(path_type=Path, exists=True), required=True,
              help="A SHARP .ply file or a folder of them.")
@click.option("--levels", default=3, show_default=True, help="Max merge levels (3 = up to 8x8 blocks).")
@click.option("--layer-gap", default=0.01, show_default=True, help="Relative depth gap below which the back layer is dropped.")
@click.option("--depth-tol", default=0.02, show_default=True, help="Max relative depth range within a merged block.")
@click.option("--color-tol", default=0.08, show_default=True, help="Max per-channel RGB range (0-1) within a merged block.")
@click.option("--min-alpha", default=0.9, show_default=True, help="Front splat must be at least this opaque to drop the one behind it.")
@click.option("--alpha-tol", default=0.1, show_default=True, help="Max opacity range within a merged block.")
@click.option("--overwrite", is_flag=True, help="Rebuild existing _quest.ply files.")
@click.option("-v", "--verbose", is_flag=True, help="Activate debug logs.")
def quest_lod_cli(input_path: Path, verbose: bool, **kwargs):
    """Build a lighter <scene>_quest.ply for standalone XR headsets."""
    logging_utils.configure(logging.DEBUG if verbose else logging.INFO)
    args = type("Args", (), kwargs)
    if input_path.is_dir():
        paths = sorted(p for p in input_path.glob("*.ply")
                       if not p.stem.endswith(("_quest", "_pruned", "_mesh")))
    else:
        paths = [input_path]
    for path in paths:
        process(path, args)
