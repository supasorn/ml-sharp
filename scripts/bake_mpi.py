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

--watch keeps running and bakes new scenes as they appear (MPI-only first, the default view, then
Hybrid-MPI; with --mpionly only MPI-only), on the GPU(s) in CUDA_VISIBLE_DEVICES; scripts/start_server.sh
runs `--watch --mpionly`.

--gpus auto picks, for each scene, the GPU with the most free memory (the machine is shared), and a scene
that runs out of GPU memory is retried up to 3 times after a short wait; --jobs sets how many run at once.

Usage: python scripts/bake_mpi.py [-i output2] [--gpus 0,1,2,3|auto] [--jobs N] [--overwrite] [--watch] [scenes...]
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
MPI_ENV = {"MPI": "1", "BINS": "128", "TILE": "16", "TEXDS": "2", "REFINE": "1", "CULL": "0.05", "CELL_BORDER": "4",
           "EXPORT_KTX2": "1", "CULL_SCALE": "0.5", "EXPORT": "_mpi", "EVAL": "0"}


def scene_plys(folder: Path, names):
    sys.path.insert(0, str(ROOT / "src"))
    from sharp.cli.bake_hybrid import scene_plys as all_plys
    plys = [p for p in all_plys(folder) if p.with_suffix(".json").exists()]
    return [p for p in plys if not names or p.stem in names]


def bake(ply: Path, gpu: str, args, log):
    out = args.input / f"{ply.stem}{'_mpionly' if args.mpionly else '_mpi'}.glb"
    if out.exists() and out.stat().st_mtime > ply.stat().st_mtime and not (args.overwrite or args.reexport):
        if not args.no_ktx2 and not out.with_name(out.stem + "_ktx2.glb").exists():
            compress(out, args, log)                          # e.g. an earlier run stopped before compressing
            return "compressed"
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
    k = glb.with_name(glb.stem + "_ktx2.glb")
    if k.exists() and k.stat().st_mtime >= glb.stat().st_mtime:   # the export already wrote it (EXPORT_KTX2)
        return
    subprocess.run([sys.executable, str(ROOT / "scripts" / "glb_ktx2.py"), str(glb),
                    str(glb.with_name(glb.stem + "_ktx2.glb"))], check=True, stdout=log, stderr=subprocess.STDOUT)


def watch(args):
    """Bake scenes that lack their MPI outputs, forever: MPI-only first (the default view), then Hybrid-MPI
    (not with --mpionly).
    A scene is picked up once its .ply and .json have been unchanged for a few seconds; a failed scene is
    retried only after its .ply changes, except that running out of GPU memory (shared GPUs) is retried up to
    6 times, waiting 30 s, 1 min, 2 min, ... up to 10 min in between (a scene too big for the GPU never fits)."""
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
    failed = {}
    oom = {}                                                   # key -> (retries so far, retry not before)
    print(f"watching {args.input} for {'MPI-only' if args.mpionly else 'MPI-only + Hybrid-MPI'} "
          f"(GPU {gpu}, every {args.poll:.0f} s)", flush=True)
    while True:
        for mpionly in ((True,) if args.mpionly else (True, False)):
            for ply in scene_plys(args.input, set()):
                sfx = "_mpionly" if mpionly else "_mpi"
                out = args.input / f"{ply.stem}{sfx}.glb"
                done_ = out.exists() and out.stat().st_mtime > ply.stat().st_mtime and (
                    args.no_ktx2 or out.with_name(out.stem + "_ktx2.glb").exists())
                key = (ply.stem, sfx)
                if done_ or failed.get(key) == ply.stat().st_mtime or oom.get(key, (0, 0))[1] > time.time():
                    continue
                if time.time() - max(ply.stat().st_mtime, ply.with_suffix(".json").stat().st_mtime) < 5:
                    continue                                   # still being written
                args.mpionly = mpionly
                t0 = time.time()
                try:
                    with open(args.work / f"{ply.stem}{'_mpionly' if mpionly else ''}.log", "w") as log:
                        status = bake(ply, gpu, args, log)
                    print(f"{time.strftime('%H:%M:%S')} {ply.stem}{sfx}: {status} in {time.time() - t0:.0f}s", flush=True)
                    if status == "skipped":                   # nothing to do after all: don't spin on it
                        failed[key] = ply.stat().st_mtime
                    oom.pop(key, None)
                except (subprocess.CalledProcessError, OSError) as e:
                    tries = oom.get(key, (0, 0))[0]
                    logf = args.work / f"{ply.stem}{'_mpionly' if mpionly else ''}.log"
                    if "OutOfMemoryError" in logf.read_text(errors="ignore") and tries < 6:
                        wait = min(30 * 2 ** tries, 600)           # shared GPUs: wait for memory, then retry
                        oom[key] = (tries + 1, time.time() + wait)
                        print(f"{time.strftime('%H:%M:%S')} {ply.stem}{sfx}: out of GPU memory, "
                              f"retry {tries + 1} in {wait} s", flush=True)
                    else:
                        failed[key] = ply.stat().st_mtime if ply.exists() else None
                        oom.pop(key, None)
                        print(f"{time.strftime('%H:%M:%S')} {ply.stem}{sfx}: FAILED ({e})", flush=True)
                break                                          # re-scan: newer scenes and MPI-only go first
            else:
                continue
            break
        else:
            time.sleep(args.poll)


def freest_gpu():
    """Index of the GPU with the most free memory (nvidia-smi)."""
    out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, check=True).stdout
    return max((int(f), i.strip()) for i, f in (l.split(",") for l in out.strip().splitlines()))[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("-i", "--input", type=Path, default=ROOT / "output2")
    ap.add_argument("--work", type=Path, default=ROOT / "renderings" / "mpi_work")
    ap.add_argument("--gpus", default="0,1,2,3", help="GPU list (one scene per GPU at a time) or 'auto'")
    ap.add_argument("--jobs", type=int, default=2, help="scenes at once with --gpus auto")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--reexport", action="store_true", help="redo only the MPI export (reuse the hybrid bakes)")
    ap.add_argument("--mpionly", action="store_true", help="MPI from all splats, no hybrid mesh (_mpionly.glb)")
    ap.add_argument("--no-ktx2", action="store_true", help="don't write the compressed-texture variant")
    ap.add_argument("--watch", action="store_true", help="keep baking new scenes as they appear")
    ap.add_argument("--poll", type=float, default=10.0, help="seconds between folder checks in --watch mode")
    ap.add_argument("scenes", nargs="*")
    args = ap.parse_args()
    args.work.mkdir(parents=True, exist_ok=True)
    if args.watch:
        return watch(args)
    plys = scene_plys(args.input, set(args.scenes))
    auto = args.gpus == "auto"
    gpus = Queue()
    for g in (["auto"] * args.jobs if auto else args.gpus.split(",")):
        gpus.put(g)
    lock = threading.Lock()
    done = [0]
    t0 = time.time()

    def job(ply):
        slot = gpus.get()
        logf = args.work / f"{ply.stem}{'_mpionly' if args.mpionly else ''}.log"
        try:
            for attempt in range(4):
                gpu = freest_gpu() if slot == "auto" else slot
                try:
                    with open(logf, "w") as log:
                        status = bake(ply, gpu, args, log)
                    break
                except subprocess.CalledProcessError as e:
                    status = f"FAILED ({e.returncode}), see {logf}"
                    if "OutOfMemoryError" not in logf.read_text(errors="ignore") or attempt == 3:
                        break
                    time.sleep(30)                         # shared GPUs: wait for memory, then retry
                    status += f" (out of GPU memory, retry {attempt + 1})"
        finally:
            gpus.put(slot)
        with lock:
            done[0] += 1
            print(f"[{done[0]}/{len(plys)} {time.time() - t0:.0f}s] {ply.stem}: {status}", flush=True)

    with ThreadPoolExecutor(args.jobs if auto else len(args.gpus.split(","))) as ex:
        list(ex.map(job, plys))


if __name__ == "__main__":
    main()
