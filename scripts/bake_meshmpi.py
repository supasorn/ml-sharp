"""Bake Mesh-MPI (viewer ?lod=meshmpi, gallery "Mesh-MPI") for scenes of a collection.

Per scene: scripts/mesh_regions.py (multi-region mesh -> <scene>_meshall.glb + its leftover splats
<scene>_prunedall.ply), then scripts/eval_alpha_patches.py bakes those splats into a sparse MPI against the mesh
(<scene>_meshmpi.glb + _astc.glb), and <scene>_meshmpi_mesh.glb links to the mesh. Work files (symlinks) go to
logs/mesh_regions/<scene>/mpiwork. Scenes that already have <scene>_meshmpi.glb are skipped unless --force.

Usage: /conda_envs/sharp_dgx/bin/python scripts/bake_meshmpi.py [scene ...] [-i data/public/output2] [--force]
       [--crack-px 3] [--fill-far-rel 0.05 --tag p]   (no scene names: every scene of the collection)
--tag t writes a variant beside the default: <scene>_mesh{all,mpi}t.glb, _prunedallt.ply, logs/mesh_regions/<scene>_t;
viewer ?lod=meshmpi-t. --fill-far-rel: see mesh_regions.py (pockets behind their surround keep their depth).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
MPI_ENV = dict(MPI="1", BINS="128", TILE="16", TEXDS="2", REFINE="1", CULL="0.05", CELL_BORDER="4", EXPORT_KTX2="1",
               KTX2_UASTC="0", KTX2_ASTC="fast", CULL_SCALE="0.25", CULL_EYES="9", KTX2_ZSTD="9", EVAL="0")


def bake(scene: str, out: Path, crack: float, tag: str = "", extra: list[str] = ()) -> None:
    t0 = time.time()
    dbg = ROOT / "logs/mesh_regions" / (scene + (f"_{tag}" if tag else ""))
    subprocess.run([PY, str(ROOT / "scripts/mesh_regions.py"), scene, "-i", str(out), "--crack-px", str(crack),
                    "--suffix", f"all{tag}", "--debug-dir", str(dbg), *extra], check=True, cwd=ROOT)
    work = dbg / "mpiwork"
    work.mkdir(parents=True, exist_ok=True)
    mpi = f"_meshmpi{tag}"
    for name, target in [(f"{scene}.ply", f"{scene}.ply"), (f"{scene}{mpi}.ply", f"{scene}_prunedall{tag}.ply"),
                         (f"{scene}{mpi}.glb", f"{scene}_meshall{tag}.glb")]:
        link = work / name
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(out / target)
    (work / f"{scene}.json").write_text((out / f"{scene}.json").read_text())
    subprocess.run([PY, str(ROOT / "scripts/eval_alpha_patches.py"), str(work / f"{scene}.ply")], check=True, cwd=ROOT,
                   env={**os.environ, **MPI_ENV, "SUFFIX": mpi, "EXPORT": mpi, "EXPORT_DIR": str(out)})
    mesh = out / f"{scene}{mpi}_mesh.glb"
    if mesh.is_symlink() or mesh.exists():
        mesh.unlink()
    mesh.symlink_to(f"{scene}_meshall{tag}.glb")
    print(f"== {scene}: Mesh-MPI done ({time.time() - t0:.0f} s)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scenes", nargs="*")
    ap.add_argument("-i", "--input", type=Path, default=ROOT / "data/public/output2")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--crack-px", type=float, default=3.0)
    ap.add_argument("--tag", default="", help="variant tag (outputs beside the default bake)")
    ap.add_argument("--fill-far-rel", type=float, default=0.0)
    args = ap.parse_args()
    out = args.input.resolve()
    scenes = args.scenes or sorted(p.stem for p in out.glob("*.json") if (out / f"{p.stem}.ply").exists())
    todo = [s for s in scenes if args.force or not (out / f"{s}_meshmpi{args.tag}.glb").exists()]
    extra = ["--fill-far-rel", str(args.fill_far_rel)] if args.fill_far_rel else []
    print(f"{len(todo)} of {len(scenes)} scenes to bake", flush=True)
    failed = []
    for i, s in enumerate(todo):
        print(f"== [{i + 1}/{len(todo)}] {s}", flush=True)
        try:
            json.loads((out / f"{s}.json").read_text())
            bake(s, out, args.crack_px, args.tag, extra)
        except Exception as e:                       # one bad scene must not stop the batch
            print(f"== {s}: FAILED ({e})", flush=True)
            failed.append(s)
    print(f"done; {len(failed)} failed: {' '.join(failed)}", flush=True)


if __name__ == "__main__":
    main()
