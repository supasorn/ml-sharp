"""Bake the Hybrid-MPI (viewer ?lod=mpi) for SHARP scenes.

Per scene <name>.ply (+ .json) in the input folder:
1. `sharp bake-hybrid --keep-move 0.3 --max-depth-ratio 0.05` into a work folder
   (not the input folder: the gallery and the bake watcher treat files there
   as scenes / outputs).
2. The hybrid's mesh is copied to <input>/<name>_mpi_mesh.glb.
3. The remaining splats become a sparse multi-plane image (128 depth bins, per-cell depth, cells
   invisible from every viewpoint in the head range dropped,
   16-px cells, scripts/eval_alpha_patches.py MPI mode) written to
   <input>/<name>_mpi.glb.
--mpionly instead builds the MPI from all splats of <name>.ply (no hybrid bake,
no mesh) and writes <input>/<name>_mpionly.glb (viewer ?lod=mpionly).
After each export, the textures are also compressed (scripts/glb_ktx2.py, Basis UASTC) into
<name>_mpi_ktx2.glb / <name>_mpionly_ktx2.glb, which the viewer loads by default (--no-ktx2 skips this).
Scenes whose output is newer than their .ply are skipped (--overwrite redoes
them, --reexport redoes only step 3, reusing the bake). Scenes run in parallel,
one per GPU.

Usage: python scripts/bake_mpi.py [-i output2] [--gpus 0,1,2,3] [--overwrite] [scene names...]
"""
import argparse
import os
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from queue import Queue

ROOT = Path(__file__).resolve().parents[1]
SUFFIX = "_hybrid_k03r05"
MPI_ENV = {"MPI": "1", "BINS": "128", "TILE": "16", "TEXDS": "2", "REFINE": "1", "CULL": "0.05", "EXPORT": "_mpi", "EVAL": "0"}


def scene_plys(folder: Path, names):
    sys.path.insert(0, str(ROOT / "src"))
    from sharp.cli.bake_hybrid import scene_plys as all_plys
    plys = [p for p in all_plys(folder) if p.with_suffix(".json").exists()]
    return [p for p in plys if not names or p.stem in names]


def bake(ply: Path, gpu: str, args, log):
    out = args.input / f"{ply.stem}{'_mpionly' if args.mpionly else '_mpi'}.glb"
    if out.exists() and out.stat().st_mtime > ply.stat().st_mtime and not (args.overwrite or args.reexport):
        return "skipped"
    work = args.work
    link = work / ply.name
    if not link.exists():
        link.symlink_to(ply.resolve())
    shutil.copy(ply.with_suffix(".json"), work / f"{ply.stem}.json")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
    if args.mpionly:
        env |= MPI_ENV | {"NOMESH": "1", "EXPORT": "_mpionly", "SUFFIX": SUFFIX, "EXPORT_DIR": str(args.input)}
        subprocess.run([sys.executable, str(ROOT / "scripts" / "eval_alpha_patches.py"), str(link)], check=True,
                       env=env, stdout=log, stderr=subprocess.STDOUT)
        compress(out, args, log)
        return "done"
    cmd = [str(Path(sys.executable).with_name("sharp")), "bake-hybrid", "-i", str(link), "--keep-move", "0.3",
           "--max-depth-ratio", "0.05", "--name-suffix", SUFFIX] + (["--overwrite"] if args.overwrite else [])
    subprocess.run(cmd, check=True, env=env, stdout=log, stderr=subprocess.STDOUT)
    shutil.copy(work / f"{ply.stem}{SUFFIX}.glb", args.input / f"{ply.stem}_mpi_mesh.glb")
    env |= MPI_ENV | {"SUFFIX": SUFFIX, "EXPORT_DIR": str(args.input)}
    subprocess.run([sys.executable, str(ROOT / "scripts" / "eval_alpha_patches.py"), str(link)], check=True,
                   env=env, stdout=log, stderr=subprocess.STDOUT)
    compress(out, args, log)
    return "done"


def compress(glb: Path, args, log):
    """<x>.glb -> <x>_ktx2.glb (compressed textures), unless --no-ktx2."""
    if args.no_ktx2:
        return
    subprocess.run([sys.executable, str(ROOT / "scripts" / "glb_ktx2.py"), str(glb),
                    str(glb.with_name(glb.stem + "_ktx2.glb"))], check=True, stdout=log, stderr=subprocess.STDOUT)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("-i", "--input", type=Path, default=ROOT / "output2")
    ap.add_argument("--work", type=Path, default=ROOT / "renderings" / "mpi_work")
    ap.add_argument("--gpus", default="0,1,2,3")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--reexport", action="store_true", help="redo only the MPI export (reuse the hybrid bakes)")
    ap.add_argument("--mpionly", action="store_true", help="MPI from all splats, no hybrid mesh (_mpionly.glb)")
    ap.add_argument("--no-ktx2", action="store_true", help="don't write the compressed-texture variant")
    ap.add_argument("scenes", nargs="*")
    args = ap.parse_args()
    args.work.mkdir(parents=True, exist_ok=True)
    plys = scene_plys(args.input, set(args.scenes))
    gpus = Queue()
    for g in args.gpus.split(","):
        gpus.put(g)
    lock = threading.Lock()
    done = [0]
    t0 = time.time()

    def job(ply):
        gpu = gpus.get()
        try:
            with open(args.work / f"{ply.stem}{'_mpionly' if args.mpionly else ''}.log", "w") as log:
                status = bake(ply, gpu, args, log)
        except subprocess.CalledProcessError as e:
            status = f"FAILED ({e.returncode}), see {args.work / (ply.stem + '.log')}"
        finally:
            gpus.put(gpu)
        with lock:
            done[0] += 1
            print(f"[{done[0]}/{len(plys)} {time.time() - t0:.0f}s] {ply.stem}: {status}", flush=True)

    with ThreadPoolExecutor(len(args.gpus.split(","))) as ex:
        list(ex.map(job, plys))


if __name__ == "__main__":
    main()
