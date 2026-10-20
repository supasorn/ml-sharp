"""Convert a 3DGS ply (e.g. from pano_predict.py) to Niantic SPZ (v2) for Spark.

~20 bytes/splat before gzip vs 56 for the ply: 24-bit fixed-point positions,
8-bit opacity/colour/log-scale, 3-byte quaternion (w implied, >= 0).

Usage:
    python scripts/pano_spz.py pano/out_ico2/hotel_room.ply   # -> hotel_room.spz
"""

from __future__ import annotations

import argparse
import gzip
import struct
from pathlib import Path

import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

FRACTIONAL_BITS = 12  # 0.24 mm steps, +-2048 m range


def u8(x):
    return np.clip(np.round(x), 0, 255).astype(np.uint8)


def convert(src: Path, dst: Path) -> int:
    from eval_alpha_patches import read_ply_vertex   # one read (plyfile goes element by element: slow on the NAS)
    return write_spz(read_ply_vertex(src), dst)


def write_spz(v, dst: Path) -> int:
    """v: the ply's vertex data (structured array with x.., f_dc_*, opacity, scale_*, rot_*)."""
    n = len(v)
    xyz = np.stack([v["x"], v["y"], v["z"]], -1).astype(np.float64)
    lim = (1 << 23) - 1
    fixed = np.clip(np.round(xyz * (1 << FRACTIONAL_BITS)), -lim, lim).astype(np.int32)
    pos = fixed.view(np.uint8).reshape(n, 3, 4)[:, :, :3].reshape(-1)  # little-endian low 3 bytes

    alpha = u8(255 / (1 + np.exp(-v["opacity"].astype(np.float64))))
    dc = np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], -1)
    color = u8(dc * (0.15 * 255) + 0.5 * 255).reshape(-1)
    scale = u8((np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], -1) + 10) * 16).reshape(-1)
    q = np.stack([v["rot_1"], v["rot_2"], v["rot_3"], v["rot_0"]], -1)  # x y z w
    q /= np.linalg.norm(q, axis=-1, keepdims=True)
    q *= np.where(q[:, 3:] < 0, -1, 1)
    rot = u8(q[:, :3] * 127.5 + 127.5).reshape(-1)

    header = struct.pack("<IIIBBBB", 0x5053474E, 2, n, 0, FRACTIONAL_BITS, 0, 0)
    with gzip.open(dst, "wb", compresslevel=1) as f:   # 1: half the time of 6, 3% larger
        for part in (header, pos, alpha, color, scale, rot):
            f.write(part.tobytes() if isinstance(part, np.ndarray) else part)
    return n


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("ply", nargs="+")
    args = p.parse_args(argv)
    for s in args.ply:
        src = Path(s)
        dst = src.with_suffix(".spz")
        n = convert(src, dst)
        print(f"{dst}: {n} splats, {dst.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
