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

--mpionly --variant ad12sep bakes the adaptive MPI-only variant (<name>_mpionly_ad12sep.glb, viewer ?lod=ad12sep).

Usage: python scripts/bake_mpi.py [-i output2] [--gpus 0,1,2,3|auto] [--jobs N] [--overwrite] [--watch]
       [--mpionly [--variant ad12sep]] [scenes...]
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
# Culling from 9 viewpoints at 0.25 resolution and Zstandard 9 for the KTX2 pages: a third of the bake time of 18 viewpoints
# at 0.5 and Zstandard 18, same quality (PSNR within 0.01 dB on two scenes), downloads ~4% larger.
# Compressed textures: plain ASTC 4x4 (<x>_astc.glb: the Quest's own format, ~1.5 s to encode); --uastc also writes
# Basis UASTC (<x>_ktx2.glb, for desktop GPUs without ASTC; ~9 s more). Without either the viewer uses the PNG file.
MPI_ENV = {"MPI": "1", "BINS": "128", "TILE": "16", "TEXDS": "2", "REFINE": "1", "CULL": "0.05", "CELL_BORDER": "4",
           "EXPORT_KTX2": "1", "KTX2_UASTC": "0", "KTX2_ASTC": "fast", "CULL_SCALE": "0.25", "CULL_EYES": "9",
           "KTX2_ZSTD": "9", "EXPORT": "_mpi", "EVAL": "0"}
# --variant (with --mpionly): extra settings of an MPI-only variant bake, written as <name>_mpionly_<variant>.glb
# (viewer ?lod=<variant>). ad12sep: 64 base planes, 3x finer where stacking shows (threshold 12) in at most 50% of
# the cells, un-premultiplied colour: about MPI-only's fill and per-fragment cost (Quest speed), +0.6-1.2 dB.
# ad12q: the earlier recipe (128 planes, 4x finer in at most 30% of the cells, colour and alpha in separate tiles):
# +2 dB but 1.6-2.1x the fill and a two-texture shader (shop scene 42 fps vs 90).
VARIANTS = {"ad12sep": {"BINS": "64", "ADAPT": "3", "ADAPT_T": "12", "ADAPT_W": "6", "ADAPT_N": "2",
                        "ADAPT_BUDGET": "0.5", "UNPREM": "1", "MAXPAGES": "6"},
            "ad12q": {"ADAPT": "4", "ADAPT_T": "12", "ADAPT_W": "6", "ADAPT_N": "2", "ADAPT_BUDGET": "0.3",
                      "UNPREM": "1", "SEPA": "1", "MAXPAGES": "6"},
            # photo / photo2: the photo's detail the splats lost, put back onto opaque surfaces (PHOTO_DETAIL), at the
            # usual 2048 px across or at the photo's own resolution (GW auto, up to 4096 px; same quad count)
            "photo": {"PHOTO_DETAIL": "1", "MAXPAGES": "8", "TEXFULL_THR": "0.0015"},
            "photo2": {"PHOTO_DETAIL": "1", "GW": "auto", "MAXPAGES": "12", "TEXFULL_THR": "0.0015"},
            # people: 4x finer planes only on the cells covering people (Mask R-CNN on the photo, grown to cover
            # the silhouette), plain MPI-only elsewhere; baked on request (gallery "People" toggle, see watch())
            "people": {"ADAPT": "4", "ADAPT_MASK": "people", "UNPREM": "1", "MAXPAGES": "6"}}
# Variants the gallery may request for a scene (<collection>/renderings/requests/<scene>.<variant>, written by web.js):
# the watcher bakes them before anything else and removes the request (renamed .failed if the bake fails).
REQUESTABLE = ("people",)
# Baked for every new scene by the MPI-only watcher (--watch --mpionly), after plain MPI-only.
WATCH_VARIANTS = ("photo",)


def scene_plys(folder: Path, names):
    """Scene .ply files (with their .json) in the folder, excluding derived outputs; as sharp.cli.bake_hybrid's
    scene_plys, which isn't imported: `sharp.cli` loads the whole SHARP model stack (~5 s)."""
    plys = sorted(p for p in folder.glob("*.ply") if not p.stem.endswith(("_quest", "_pruned", "_mesh", "_hybrid")))
    return [p for p in plys if p.with_suffix(".json").exists() and (not names or p.stem in names)]


class Worker:
    """A persistent `eval_alpha_patches.py --serve` process (one per GPU and settings): Python, torch, CUDA and gsplat
    load once instead of per scene (~5 s each), and a scene's files are written in the background while the next
    scene runs on the GPU. It answers "gpu-done <ply>", then "ok <ply>" (or "fail <ply> ...") per scene."""
    def __init__(self, env):
        self.p = subprocess.Popen([sys.executable, str(ROOT / "scripts" / "eval_alpha_patches.py"), "--serve"], env=env,
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
        self.lock, self.waiting = threading.Lock(), {}         # ply -> (gpu-done Event, done Event, [result])
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        for line in self.p.stdout:
            kind, _, rest = line.rstrip("\n").partition(" ")
            # the ply path may contain spaces ("x~2 (2).ply"): match it against the scenes being waited for
            # (splitting at the first space lost such a scene's answer and its bake waited forever)
            with self.lock:
                ply = next((k for k in self.waiting if rest == k or rest.startswith(k + " ")), None)
                w = self.waiting.get(ply)
            if w is None:
                continue
            msg = rest[len(ply) + 1:]
            if kind == "gpu-done":
                w[0].set()
            else:
                w[2].append(None if kind == "ok" else msg or kind)
                w[0].set(); w[1].set()
        with self.lock:                                   # the worker exited: fail whatever is pending
            for w in self.waiting.values():
                w[2].append("bake worker exited"); w[0].set(); w[1].set()

    def bake(self, ply, log_path, gpu_done=None):
        w = (threading.Event(), threading.Event(), [])
        with self.lock:
            self.waiting[str(ply)] = w
            self.p.stdin.write(f"{ply}\t{log_path}\n"); self.p.stdin.flush()
        w[0].wait()
        if gpu_done is not None:
            gpu_done()                                    # the GPU is free for the next scene
        w[1].wait()
        with self.lock:
            self.waiting.pop(str(ply), None)
        if w[2] and w[2][0] is not None:
            raise subprocess.CalledProcessError(1, f"bake worker: {w[2][0]}")


WORKERS, WORKERS_LOCK = {}, threading.Lock()


def worker_bake(env, ply, log, gpu_done=None):
    """Bake one scene with the worker for these settings (started on first use, restarted if it died); gpu_done() is
    called once its GPU work is done (its files are still being written)."""
    key = tuple(sorted(env.items()))
    with WORKERS_LOCK:
        w = WORKERS.get(key)
        if w is None or w.p.poll() is not None:
            w = WORKERS[key] = Worker(env)
    log.flush()
    w.bake(ply, log.name, gpu_done)


def compressed(glb: Path):
    """Its compressed-texture copy (ASTC or UASTC) exists and is current."""
    return any(k.exists() and k.stat().st_mtime >= glb.stat().st_mtime
               for k in (glb.with_name(glb.stem + "_astc.glb"), glb.with_name(glb.stem + "_ktx2.glb")))


def out_suffix(args, mpionly):
    return ("_mpionly" + (f"_{args.variant}" if args.variant else "")) if mpionly else "_mpi"


def bake(ply: Path, gpu: str, args, log, gpu_done=None):
    out = args.input / f"{ply.stem}{out_suffix(args, args.mpionly)}.glb"
    if out.exists() and out.stat().st_mtime > ply.stat().st_mtime and not (args.overwrite or args.reexport):
        if not args.no_ktx2 and not compressed(out):
            compress(out, args, log)                          # e.g. an earlier run stopped before compressing
            return "compressed"
        return "skipped"
    work = args.work
    link = work / ply.name
    if not link.exists():
        link.symlink_to(ply.resolve())
    shutil.copy(ply.with_suffix(".json"), work / f"{ply.stem}.json")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu) | ({"KTX2_UASTC": "1"} if args.uastc else {})
    if args.mpionly:
        env |= MPI_ENV | {"NOMESH": "1", "EXPORT": out_suffix(args, True), "SUFFIX": SUFFIX,
                          "EXPORT_DIR": str(args.input)} | VARIANTS.get(args.variant, {})
        if args.uastc:
            env["KTX2_UASTC"] = "1"
        worker_bake(env, link, log, gpu_done)
        compress(out, args, log)
        return "done"
    cmd = [str(Path(sys.executable).with_name("sharp")), "bake-hybrid", "-i", str(link), "--keep-move", "0.3",
           "--max-depth-ratio", "0.05", "--name-suffix", SUFFIX] + (["--overwrite"] if args.overwrite else [])
    subprocess.run(cmd, check=True, env=env, stdout=log, stderr=subprocess.STDOUT)
    shutil.copy(work / f"{ply.stem}{SUFFIX}.glb", args.input / f"{ply.stem}_mpi_mesh.glb")
    env |= MPI_ENV | {"SUFFIX": SUFFIX, "EXPORT_DIR": str(args.input)} | ({"KTX2_UASTC": "1"} if args.uastc else {})
    subprocess.run([sys.executable, str(ROOT / "scripts" / "eval_alpha_patches.py"), str(link)], check=True,
                   env=env, stdout=log, stderr=subprocess.STDOUT)
    compress(out, args, log)
    return "done"


def compress(glb: Path, args, log):
    """<x>.glb -> <x>_ktx2.glb (compressed textures), unless --no-ktx2 or the export already wrote a compressed copy."""
    if args.no_ktx2 or compressed(glb):
        return
    subprocess.run([sys.executable, str(ROOT / "scripts" / "glb_ktx2.py"), str(glb),
                    str(glb.with_name(glb.stem + "_ktx2.glb"))], check=True, stdout=log, stderr=subprocess.STDOUT)


def watch(args):
    """Bake scenes that lack their MPI outputs, forever: MPI-only first, then (--mpionly) the WATCH_VARIANTS (photo:
    the viewer's default view; MPI-only is its fallback until then), or (without --mpionly) Hybrid-MPI.
    A scene is picked up once its .ply and .json have been unchanged for a few seconds; a failed scene is
    retried only after its .ply changes, except that running out of GPU memory (shared GPUs) is retried up to
    6 times, waiting 30 s, 1 min, 2 min, ... up to 10 min in between (a scene too big for the GPU never fits)."""
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
    failed = {}
    oom = {}                                                   # key -> (retries so far, retry not before)
    base = args.variant
    stages = ([(True, base)] + [(True, v) for v in WATCH_VARIANTS if v != base]) if args.mpionly \
        else [(True, base), (False, base)]
    print(f"watching {args.input} for "
          f"{' + '.join(('MPI-only' + (f' {v}' if v else '')) if m else 'Hybrid-MPI' for m, v in stages)} "
          f"(GPU {gpu}, every {args.poll:.0f} s)", flush=True)
    reqdir = args.work.parent / "requests"
    while True:
        if args.mpionly and reqdir.is_dir():
            for req in sorted(reqdir.iterdir(), key=lambda q: q.stat().st_mtime):
                name, _, variant = req.name.rpartition(".")
                ply = args.input / f"{name}.ply"
                if variant not in REQUESTABLE or not ply.exists():
                    continue
                t0 = time.time()
                saved = args.variant, args.overwrite
                args.variant, args.overwrite = variant, True
                try:
                    with open(args.work / f"{name}{out_suffix(args, True)}.log", "w") as log:
                        status = bake(ply, gpu, args, log)
                    req.unlink(missing_ok=True)
                    print(f"{time.strftime('%H:%M:%S')} {name} {variant} (requested): {status} in {time.time() - t0:.0f}s",
                          flush=True)
                except (subprocess.CalledProcessError, OSError) as e:
                    req.replace(req.with_name(req.name + ".failed"))
                    print(f"{time.strftime('%H:%M:%S')} {name} {variant} (requested): FAILED ({e})", flush=True)
                finally:
                    args.variant, args.overwrite = saved
        for mpionly, variant in stages:
            args.variant = variant
            for ply in scene_plys(args.input, set()):
                sfx = out_suffix(args, mpionly)
                out = args.input / f"{ply.stem}{sfx}.glb"
                done_ = out.exists() and out.stat().st_mtime > ply.stat().st_mtime and (
                    args.no_ktx2 or compressed(out))
                key = (ply.stem, sfx)
                if done_ or failed.get(key) == ply.stat().st_mtime or oom.get(key, (0, 0))[1] > time.time():
                    continue
                if time.time() - max(ply.stat().st_mtime, ply.with_suffix(".json").stat().st_mtime) < 5:
                    continue                                   # still being written
                args.mpionly = mpionly
                t0 = time.time()
                try:
                    with open(args.work / f"{ply.stem}{out_suffix(args, True) if mpionly else ''}.log", "w") as log:
                        status = bake(ply, gpu, args, log)
                    print(f"{time.strftime('%H:%M:%S')} {ply.stem}{sfx}: {status} in {time.time() - t0:.0f}s", flush=True)
                    if status == "skipped":                   # nothing to do after all: don't spin on it
                        failed[key] = ply.stat().st_mtime
                    oom.pop(key, None)
                except (subprocess.CalledProcessError, OSError) as e:
                    tries = oom.get(key, (0, 0))[0]
                    logf = args.work / f"{ply.stem}{out_suffix(args, True) if mpionly else ''}.log"
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
    ap.add_argument("--uastc", action="store_true", help="also write Basis UASTC textures (<x>_ktx2.glb, desktop GPUs)")
    ap.add_argument("--variant", choices=sorted(VARIANTS), help="MPI-only variant bake (with --mpionly; see VARIANTS)")
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
        held = [True]

        def release():                                    # MPI-only: the GPU is free once the scene's GPU work is done
            if held[0]:
                held[0] = False
                gpus.put(slot)
        logf = args.work / f"{ply.stem}{out_suffix(args, True) if args.mpionly else ''}.log"
        try:
            for attempt in range(4):
                gpu = freest_gpu() if slot == "auto" else slot
                try:
                    with open(logf, "w") as log:
                        status = bake(ply, gpu, args, log, release)
                    break
                except subprocess.CalledProcessError as e:
                    status = f"FAILED ({e.returncode}), see {logf}"
                    if "OutOfMemoryError" not in logf.read_text(errors="ignore") or attempt == 3:
                        break
                    time.sleep(30)                         # shared GPUs: wait for memory, then retry
                    status += f" (out of GPU memory, retry {attempt + 1})"
        finally:
            release()
        with lock:
            done[0] += 1
            print(f"[{done[0]}/{len(plys)} {time.time() - t0:.0f}s] {ply.stem}: {status}", flush=True)

    # MPI-only: twice the GPU slots, so a scene's files are written while the next one runs on the GPU
    slots = args.jobs if auto else len(args.gpus.split(","))
    with ThreadPoolExecutor(slots * (2 if args.mpionly else 1)) as ex:
        list(ex.map(job, plys))


if __name__ == "__main__":
    main()
